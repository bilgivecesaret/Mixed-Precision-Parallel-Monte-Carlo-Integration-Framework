#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cmath>
#include <chrono>
#include <random>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

enum class TokenType { Number, Operator, Function, Variable };

enum OperatorID {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW,
    FUNC_SIN, FUNC_COS, FUNC_LOG, FUNC_LN, FUNC_EXP, FUNC_SQRT,
    VAR_X, VAR_Y
};

struct Token {
    TokenType type;
    union {
        float number_value;
        int operator_id;
    };

    __host__ __device__ Token() {}
    __host__ __device__ Token(TokenType t, float num) : type(t), number_value(num) {}
    __host__ __device__ Token(TokenType t, int id) : type(t), operator_id(id) {}
};


std::vector<Token> tokenize(const std::string& expr) {
    std::vector<Token> tokens;
    size_t i = 0;
    
    while(i < expr.size()) {
        char c = expr[i];
        if(isspace(c)) { ++i; continue; }

        if(isdigit(c) || c == '.' || c == 'e' || c == 'E') {
            std::string num;
            bool has_exp = false;
            while(i < expr.size() && (isdigit(expr[i]) || expr[i] == '.' || 
                  (tolower(expr[i]) == 'e' && !has_exp))) {
                if(tolower(expr[i]) == 'e') {
                    has_exp = true;
                    num += expr[i++];
                    if(i < expr.size() && (expr[i] == '+' || expr[i] == '-')) 
                        num += expr[i++];
                } else {
                    num += expr[i++];
                }
            }
            tokens.emplace_back(TokenType::Number, std::stof(num));
        }
        else if(isalpha(c)) {
            std::string name;
            while(i < expr.size() && isalpha(expr[i])) name += expr[i++];
            
            if(name == "sin") tokens.emplace_back(TokenType::Function, FUNC_SIN);
            else if(name == "cos") tokens.emplace_back(TokenType::Function, FUNC_COS);
            else if(name == "log") tokens.emplace_back(TokenType::Function, FUNC_LOG);
            else if(name == "ln") tokens.emplace_back(TokenType::Function, FUNC_LN);
            else if(name == "exp") tokens.emplace_back(TokenType::Function, FUNC_EXP);
            else if(name == "sqrt") tokens.emplace_back(TokenType::Function, FUNC_SQRT);
            else if(name == "x") tokens.emplace_back(TokenType::Variable, VAR_X);
            else if(name == "y") tokens.emplace_back(TokenType::Variable, VAR_Y);
            else throw std::runtime_error("Unknown function/variable: " + name);
        }
        else if(std::string("+-*/^").find(c) != std::string::npos) {
            int op_id;
            if(c == '+') op_id = OP_ADD;
            else if(c == '-') op_id = OP_SUB;
            else if(c == '*') op_id = OP_MUL;
            else if(c == '/') op_id = OP_DIV;
            else if(c == '^') op_id = OP_POW;
            tokens.emplace_back(TokenType::Operator, op_id);
            ++i;
        }
        else if(c == '(' || c == ')') {
            tokens.emplace_back(TokenType::Operator, (c == '(') ? -1 : -2);
            ++i;
        }
        else {
            throw std::runtime_error(std::string("Invalid character: ") + c);
        }
    }
    return tokens;
}

std::vector<Token> to_postfix(const std::vector<Token>& tokens) {
    std::vector<Token> output;
    std::stack<Token> stack;

    for(const auto& token : tokens) {
        if(token.type == TokenType::Number || token.type == TokenType::Variable) {
            output.push_back(token);
        }
        else if(token.type == TokenType::Function) {
            stack.push(token);
        }
        else if(token.type == TokenType::Operator) {
            if(token.operator_id == -1) {
                stack.push(token);
            }
            else if(token.operator_id == -2) {
                while(!stack.empty() && stack.top().operator_id != -1) {
                    output.push_back(stack.top());
                    stack.pop();
                }
                if(stack.empty()) throw std::runtime_error("Mismatched parentheses");
                stack.pop();
            }
            else {
                while(!stack.empty() && stack.top().type == TokenType::Operator &&
                      stack.top().operator_id != -1 && stack.top().operator_id != -2 &&
                      ((stack.top().operator_id >= OP_ADD && stack.top().operator_id <= OP_POW) || 
                       (stack.top().operator_id >= FUNC_SIN && stack.top().operator_id <= FUNC_SQRT))) {
                    output.push_back(stack.top());
                    stack.pop();
                }
                stack.push(token);
            }
        }
    }

    while(!stack.empty()) {
        output.push_back(stack.top());
        stack.pop();
    }
    return output;
}

float evaluate_postfix(const std::vector<Token>& postfix, float x, float y) {
    std::stack<float> stack;
    
    for(const auto& token : postfix) {
        if(token.type == TokenType::Number) {
            stack.push(token.number_value);
        }
        else if(token.type == TokenType::Variable) {
            stack.push((token.operator_id == VAR_X) ? x : y);
        }
        else if(token.type == TokenType::Operator) {
            float b = stack.top(); stack.pop();
            float a = stack.top(); stack.pop();
            
            switch(token.operator_id) {
                case OP_ADD: stack.push(a + b); break;
                case OP_SUB: stack.push(a - b); break;
                case OP_MUL: stack.push(a * b); break;
                case OP_DIV: stack.push(a / b); break;
                case OP_POW: stack.push(pow(a, b)); break;
            }
        }
        else if(token.type == TokenType::Function) {
            float a = stack.top(); stack.pop();
            switch(token.operator_id) {
                case FUNC_SIN: stack.push(sin(a)); break;
                case FUNC_COS: stack.push(cos(a)); break;
                case FUNC_LOG: stack.push(log10(a)); break;
                case FUNC_LN: stack.push(log(a)); break;
                case FUNC_EXP: stack.push(exp(a)); break;
                case FUNC_SQRT: stack.push(sqrt(a)); break;
            }
        }
    }
    return stack.top();
}

__device__ float evaluate_postfix_device(const Token* postfix, int size, float x, float y) {
    float stack[64];
    int sp = 0;

    for(int i=0; i<size; ++i) {
        Token token = postfix[i];
        if(token.type == TokenType::Number) {
            stack[sp++] = token.number_value;
        }
        else if(token.type == TokenType::Variable) {
            stack[sp++] = (token.operator_id == VAR_X) ? x : y;
        }
        else if(token.type == TokenType::Operator) {
            float b = stack[--sp];
            float a = stack[--sp];
            
            switch(token.operator_id) {
                case OP_ADD: stack[sp++] = a + b; break;
                case OP_SUB: stack[sp++] = a - b; break;
                case OP_MUL: stack[sp++] = a * b; break;
                case OP_DIV: stack[sp++] = a / b; break;
                case OP_POW: stack[sp++] = powf(a, b); break;
            }
        }
        else if(token.type == TokenType::Function) {
            float a = stack[--sp];
            switch(token.operator_id) {
                case FUNC_SIN: stack[sp++] = sinf(a); break;
                case FUNC_COS: stack[sp++] = cosf(a); break;
                case FUNC_LOG: stack[sp++] = log10f(a); break;
                case FUNC_LN: stack[sp++] = logf(a); break;
                case FUNC_EXP: stack[sp++] = expf(a); break;
                case FUNC_SQRT: stack[sp++] = sqrtf(a); break;
            }
        }
    }
    return stack[0];
}

__global__ void monte_carlo_kernel(float* results, size_t samples, float a, float b, 
                                  float c, float d, Token* postfix, int postfix_size, unsigned seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= samples) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    
    float x = a + (b-a)*curand_uniform(&state);
    float y = c + (d-c)*curand_uniform(&state);
    
    results[idx] = evaluate_postfix_device(postfix, postfix_size, x, y);
}

float parallel_integrate(size_t samples, float a, float b, float c, float d,
                        const std::vector<Token>& postfix, float& time_ms) {
    Token* d_postfix;
    float* d_results;

    cudaEvent_t start, stop;

    dim3 block(256);
    dim3 grid((samples + block.x - 1)/block.x);
    monte_carlo_kernel<<<grid, block>>>(d_results, samples, a, b, c, d, d_postfix, postfix.size(), time(nullptr));

    std::vector<float> results(samples);

    float sum = 0;
    for(float val : results) sum += val;
    float area = (b-a)*(d-c);

    return area * (sum/samples);
}

float serial_integrate(size_t samples, float a, float b, float c, float d,
                      const std::vector<Token>& postfix, float& time_ms) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_x(a, b);
    std::uniform_real_distribution<float> dist_y(c, d);

    float sum = 0;
    for(size_t i=0; i<samples; ++i) {
        float x = dist_x(gen);
        float y = dist_y(gen);
        sum += evaluate_postfix(postfix, x, y);
    }
    float area = (b-a)*(d-c);
    
    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    return area * (sum/samples);
}

int main() {
    std::string expr;
    float a, b, c, d;
    size_t samples;

    std::cout << "Enter function (e.g., x*y + sin(x)): ";
    std::getline(std::cin, expr);
    std::cout << "Enter a b c d: ";
    std::cin >> a >> b >> c >> d;
    std::cout << "Number of samples: ";
    std::cin >> samples;

    auto tokens = tokenize(expr);
    auto postfix = to_postfix(tokens);

    float parallel_time, serial_time;
    float parallel_result = parallel_integrate(samples, a, b, c, d, postfix, parallel_time);
    float serial_result = serial_integrate(samples, a, b, c, d, postfix, serial_time);

    std::cout << "\n=== Results ===\n";
    std::cout << "Parallel result: " << parallel_result << " (" << parallel_time << " ms)\n";
    std::cout << "Serial result:   " << serial_result << " (" << serial_time << " ms)\n";
    std::cout << "Speedup: " << serial_time/parallel_time << "x\n";

    return 0;
}