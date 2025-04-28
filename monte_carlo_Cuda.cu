#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cmath>
#include <cctype>
#include <random>
#include <chrono>
#include <limits>
#include <sstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

enum class TokenType { Number, Operator, Function, Variable, LeftParen, RightParen };

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

// Helper function for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::vector<Token> tokenize(const std::string& expr) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < expr.length()) {
        char c = expr[i];

        if (isspace(c)) { ++i; continue; }

        if (isdigit(c) || c == '.') {
            std::string num;
            while (i < expr.length() && (isdigit(expr[i]) || expr[i] == '.')) num += expr[i++];
            tokens.emplace_back(TokenType::Number, num);
        }
        else if (isalpha(c)) {
            std::string name;
            while (i < expr.length() && isalpha(expr[i])) name += expr[i++];
            if (name == "sin" || name == "cos" || name == "log" || name == "ln" || name == "exp" || name == "sqrt")
                tokens.emplace_back(TokenType::Function, name);
            else
                tokens.emplace_back(TokenType::Variable, name);
        }
        else if (std::string("+-*/^").find(c) != std::string::npos) {
            // Handle unary minus/plus
            if ((c == '-' || c == '+') && (i == 0 || 
                (i > 0 && (expr[i-1] == '(' || 
                 std::string("+-*/^").find(expr[i-1]) != std::string::npos)))) {
                
                if (c == '-') {
                    // Insert 0 before unary minus to handle it as binary operation (0-x)
                    tokens.emplace_back(TokenType::Number, "0");
                }
                // For unary plus, we can just skip it
                if (c == '-') {
                    tokens.emplace_back(TokenType::Operator, std::string(1, c));
                }
            } else {
                tokens.emplace_back(TokenType::Operator, std::string(1, c));
            }
            ++i;
        }
        else if (c == '(') {
            tokens.emplace_back(TokenType::LeftParen, "(");
            ++i;
        }
        else if (c == ')') {
            tokens.emplace_back(TokenType::RightParen, ")");
            ++i;
        }
        else {
            throw std::runtime_error(std::string("Invalid character: ") + c);
        }
    }

    return tokens;
}

int precedence(const std::string& op) {
    if (op == "+" || op == "-") return 1;
    if (op == "*" || op == "/") return 2;
    if (op == "^") return 3;
    return 0;
}

bool is_right_associative(const std::string& op) {
    return op == "^";
}

std::vector<Token> to_postfix(const std::vector<Token>& tokens) {
    std::vector<Token> output;
    std::stack<Token> stack;

    for (const auto& token : tokens) {
        if (token.type == TokenType::Number || token.type == TokenType::Variable) {
            output.push_back(token);
        }
        else if (token.type == TokenType::Function) {
            stack.push(token);
        }
        else if (token.type == TokenType::Operator) {
            while (!stack.empty() && stack.top().type == TokenType::Operator &&
                   ((precedence(stack.top().value) > precedence(token.value)) ||
                    (precedence(stack.top().value) == precedence(token.value) &&
                     !is_right_associative(token.value)))) {
                output.push_back(stack.top());
                stack.pop();
            }
            stack.push(token);
        }
        else if (token.type == TokenType::LeftParen) {
            stack.push(token);
        }
        else if (token.type == TokenType::RightParen) {
            while (!stack.empty() && stack.top().type != TokenType::LeftParen) {
                output.push_back(stack.top());
                stack.pop();
            }
            if (stack.empty()) throw std::runtime_error("Mismatched parentheses");
            stack.pop();
            if (!stack.empty() && stack.top().type == TokenType::Function) {
                output.push_back(stack.top());
                stack.pop();
            }
        }
    }

    while (!stack.empty()) {
        if (stack.top().type == TokenType::LeftParen) throw std::runtime_error("Mismatched parentheses");
        output.push_back(stack.top());
        stack.pop();
    }

    return output;
}

template<typename T>
T evaluate_postfix(const std::vector<Token>& postfix, T x_val, T y_val) {
    std::stack<T> stack;
    for (const auto& token : postfix) {
        if (token.type == TokenType::Number) {
            stack.push(static_cast<T>(std::stold(token.value)));
        } else if (token.type == TokenType::Variable) {
            if (token.value == "x") stack.push(x_val);
            else if (token.value == "y") stack.push(y_val);
            else throw std::runtime_error("Unknown variable: " + token.value);
        } else if (token.type == TokenType::Operator) {
            T b = stack.top(); stack.pop();
            T a = stack.top(); stack.pop();
            if (token.value == "+") stack.push(a + b);
            else if (token.value == "-") stack.push(a - b);
            else if (token.value == "*") stack.push(a * b);
            else if (token.value == "/") { if (b == 0) throw std::runtime_error("Div by 0"); stack.push(a / b); }
            else if (token.value == "^") stack.push(std::pow(a, b));
        } else if (token.type == TokenType::Function) {
            T a = stack.top(); stack.pop();
            if (token.value == "sin") stack.push(std::sin(a));
            else if (token.value == "cos") stack.push(std::cos(a));
            else if (token.value == "log") { if (a <= 0) throw std::runtime_error("Log domain"); stack.push(std::log10(a)); }
            else if (token.value == "ln") { if (a <= 0) throw std::runtime_error("Ln domain"); stack.push(std::log(a)); }
            else if (token.value == "exp") stack.push(std::exp(a));
            else if (token.value == "sqrt") { if (a < 0) throw std::runtime_error("Sqrt domain"); stack.push(std::sqrt(a)); }
        }
    }
    return stack.top();
}

enum class Precision { Float, Double, LongDouble };

Precision select_precision(long double avg, long double grad, long double var, double tol) {
    const long double eps_float = std::numeric_limits<float>::epsilon();
    const long double eps_double = std::numeric_limits<double>::epsilon();
    const long double eps_long_double = std::numeric_limits<long double>::epsilon();

    // Estimate max function value
    long double max_val = std::max(std::abs(avg + std::sqrt(var)), std::abs(avg - std::sqrt(var)));

    // Normalize: To avoid distorting the error estimate by using very small values.
    max_val = std::max(max_val, static_cast<long double>(tol));  // Don't let very small values distort the error estimate.
    grad = std::max(grad, static_cast<long double>(tol));    // If the gradient is almost zero, the minimum error estimate can still be made.

    // Estimate error as epsilon * max gradient * scale
    long double error_float = eps_float * max_val * grad;
    long double error_double = eps_double * max_val * grad;
    long double error_long_double = eps_long_double * max_val * grad;

    // Compare estimated relative error with tolerance
    if (error_float <= tol) return Precision::Float;
    if (error_double <= tol) return Precision::Double;
    return Precision::LongDouble;
}

// Custom device string comparison function
__device__ bool device_strcmp(const char* str1, const char* str2) {
    while (*str1 != '\0' && *str2 != '\0') {
        if (*str1 != *str2) return false;
        ++str1;
        ++str2;
    }
    return *str1 == '\0' && *str2 == '\0';
}

// CUDA device function for evaluating postfix expression
template<typename T>
__device__ T evaluate_postfix_device(const TokenType* types, const char* values, int token_count, T x_val, T y_val) {
    T stack[100]; // Fixed-size stack for GPU
    int stack_pos = 0;

    for (int i = 0; i < token_count; ++i) {
        TokenType type = types[i];

        if (type == TokenType::Number) {
            // Parse number from string
            T num = 0;
            bool decimal = false;
            T decimal_pos = 1;

            for (int j = 0; values[i * 20 + j] != '\0' && j < 20; ++j) {
                char c = values[i * 20 + j];
                if (c == '.') {
                    decimal = true;
                } else if (c >= '0' && c <= '9') {
                    if (decimal) {
                        decimal_pos *= 0.1;
                        num += (c - '0') * decimal_pos;
                    } else {
                        num = num * 10 + (c - '0');
                    }
                }
            }
            stack[stack_pos++] = num;
        } else if (type == TokenType::Variable) {
            char var = values[i * 20];
            if (var == 'x') stack[stack_pos++] = x_val;
            else if (var == 'y') stack[stack_pos++] = y_val;
            // Ignore other variables
        } else if (type == TokenType::Operator) {
            char op = values[i * 20];
            T b = stack[--stack_pos];
            T a = stack[--stack_pos];

            if (op == '+') stack[stack_pos++] = a + b;
            else if (op == '-') stack[stack_pos++] = a - b;
            else if (op == '*') stack[stack_pos++] = a * b;
            else if (op == '/') {
                if (b == 0) return 0; // Error handling: return 0 for division by zero
                stack[stack_pos++] = a / b;
            } else if (op == '^') stack[stack_pos++] = pow(a, b);
        } else if (type == TokenType::Function) {
            T a = stack[--stack_pos];

            // Check first three chars to determine function
            const char* func = &values[i * 20];

            if (device_strcmp(func, "sin")) stack[stack_pos++] = sin(a);
            else if (device_strcmp(func, "cos")) stack[stack_pos++] = cos(a);
            else if (device_strcmp(func, "log")) {
                if (a <= 0) return 0; // Error handling
                stack[stack_pos++] = log10(a);
            } else if (device_strcmp(func, "ln")) {
                if (a <= 0) return 0; // Error handling
                stack[stack_pos++] = log(a);
            } else if (device_strcmp(func, "exp")) stack[stack_pos++] = exp(a);
            else if (device_strcmp(func, "sqrt")) {
                if (a < 0) return 0; // Error handling
                stack[stack_pos++] = sqrt(a);
            }
        }
    }

    return stack_pos > 0 ? stack[stack_pos - 1] : 0;
}

// CUDA kernel for Monte Carlo integration
template <typename T>
__global__ void monte_carlo_kernel(TokenType* types, char* values, int token_count, 
                                  T a, T b, T c, T d, 
                                  unsigned long long int samples_per_thread, 
                                  T* results, unsigned long long int* valid_counts,
                                  unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random number generator
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    
    T sum = 0;
    unsigned long long int valid = 0;
    
    for (unsigned long long int i = 0; i < samples_per_thread; ++i) {
        // Generate random points in the integration domain
        T x = a + (b - a) * curand_uniform(&state);
        T y = c + (d - c) * curand_uniform(&state);
        
        // Evaluate function
        T value = evaluate_postfix_device<T>(types, values, token_count, x, y);
        
        // Check for NaN or Inf
        if (isfinite(value)) {
            sum += value;
            valid++;
        }
    }
    
    // Store results
    results[idx] = sum;
    valid_counts[idx] = valid;
}

// Helper function to prepare CUDA data from tokens
void prepare_cuda_data(const std::vector<Token>& postfix, TokenType** d_types, char** d_values) {
    int token_count = postfix.size();
    
    // Allocate host memory
    TokenType* h_types = new TokenType[token_count];
    char* h_values = new char[token_count * 20]; // Assume max 20 chars per token value
    
    // Fill host data
    for (int i = 0; i < token_count; ++i) {
        h_types[i] = postfix[i].type;

        // Copy value with zero padding
        strncpy(&h_values[i * 20], postfix[i].value.c_str(), 19); // Removed std::
        h_values[i * 20 + std::min(static_cast<int>(postfix[i].value.length()), 19)] = '\0';
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(d_types, token_count * sizeof(TokenType)));
    CUDA_CHECK(cudaMalloc(d_values, token_count * 20 * sizeof(char)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(*d_types, h_types, token_count * sizeof(TokenType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_values, h_values, token_count * 20 * sizeof(char), cudaMemcpyHostToDevice));
    
    // Free host memory
    delete[] h_types;
    delete[] h_values;
}

template<typename T>
T monte_carlo_integrate_2d_cuda(size_t samples, T a, T b, T c, T d, const std::vector<Token>& postfix) {
    // Determine number of CUDA threads and blocks
    int threadsPerBlock = 256;
    int blocks = std::min(65535, static_cast<int>((samples + threadsPerBlock - 1) / threadsPerBlock));
    unsigned long long int samples_per_thread = (samples + blocks * threadsPerBlock - 1) / (blocks * threadsPerBlock);
    
    // Prepare CUDA data
    TokenType* d_types = nullptr;
    char* d_values = nullptr;
    prepare_cuda_data(postfix, &d_types, &d_values);
    
    // Allocate result arrays
    T* d_results = nullptr;
    unsigned long long int* d_valid_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, blocks * threadsPerBlock * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_valid_counts, blocks * threadsPerBlock * sizeof(unsigned long long int)));
    
    // Launch kernel
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    monte_carlo_kernel<T><<<blocks, threadsPerBlock>>>(d_types, d_values, postfix.size(), 
                                                     a, b, c, d, samples_per_thread, 
                                                     d_results, d_valid_counts, seed);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    T* h_results = new T[blocks * threadsPerBlock];
    unsigned long long int* h_valid_counts = new unsigned long long int[blocks * threadsPerBlock];
    CUDA_CHECK(cudaMemcpy(h_results, d_results, blocks * threadsPerBlock * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid_counts, d_valid_counts, blocks * threadsPerBlock * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    
    // Aggregate results
    T sum = 0;
    unsigned long long int valid_total = 0;
    for (int i = 0; i < blocks * threadsPerBlock; ++i) {
        sum += h_results[i];
        valid_total += h_valid_counts[i];
    }
    
    // Clean up
    delete[] h_results;
    delete[] h_valid_counts;
    CUDA_CHECK(cudaFree(d_types));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_valid_counts));
    
    if (valid_total == 0) return 0;
    
    // Calculate result
    T area = (b - a) * (d - c);
    return area * sum / valid_total;
}

long double average_value(const std::vector<Token>& postfix, long double a, long double b, long double c, long double d) {
    const int N = 10;
    long double sum = 0;
    int valid = 0;

    // We'll use CPU for this analysis step
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            long double x = a + (b - a) * i / (N - 1.0);
            long double y = c + (d - c) * j / (N - 1.0);
            try {
                sum += std::abs(evaluate_postfix<long double>(postfix, x, y));
                valid++;
            } catch (...) {}
        }
    }
    return valid ? sum / valid : 0;
}

long double estimate_variance(const std::vector<Token>& postfix, long double a, long double b, long double c, long double d, size_t samples) {
    samples = std::min(samples, static_cast<size_t>(1000));
    long double sum = 0, sum_sq = 0;
    int valid = 0;

    // Using CPU for this analysis step
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<long double> dist_x(a, b);
    std::uniform_real_distribution<long double> dist_y(c, d);

    for (size_t i = 0; i < samples; ++i) {
        long double x = dist_x(gen);
        long double y = dist_y(gen);
        try {
            long double val = evaluate_postfix<long double>(postfix, x, y);
            sum += val;
            sum_sq += val * val;
            valid++;
        } catch (...) {}
    }
    
    if (valid < 2) return 0;
    long double mean = sum / valid;
    return (sum_sq / valid) - (mean * mean);
}

long double estimate_gradient(const std::vector<Token>& postfix, long double a, long double b, long double c, long double d) {
    const int N = 10;
    long double max_grad = 0;

    // Using CPU for this analysis step
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            long double x = a + (b - a) * i / (N - 1.0);
            long double y = c + (d - c) * j / (N - 1.0);
            try {
                long double fx = evaluate_postfix<long double>(postfix, x, y);
                if (i < N-1) {
                    long double x_next = a + (b - a) * (i+1) / (N - 1.0);
                    long double fx_next = evaluate_postfix<long double>(postfix, x_next, y);
                    long double grad_x = std::abs((fx_next - fx) / (x_next - x));
                    max_grad = std::max(max_grad, grad_x);
                }
                if (j < N-1) {
                    long double y_next = c + (d - c) * (j+1) / (N - 1.0);
                    long double fy_next = evaluate_postfix<long double>(postfix, x, y_next);
                    long double grad_y = std::abs((fy_next - fx) / (y_next - y));
                    max_grad = std::max(max_grad, grad_y);
                }
            } catch (...) {}
        }
    }
    return max_grad;
}

std::vector<std::string> split_expression(const std::string& expr) {
    std::vector<std::string> terms;
    std::string current;
    int paren = 0;
    
    for (size_t i = 0; i < expr.length(); ++i) {
        char c = expr[i];
        
        if (c == '(') paren++;
        else if (c == ')') paren--;
        
        // Check for +/- operators that are not inside parentheses
        if ((c == '+' || c == '-') && paren == 0) {
            // Skip if this is the first character or after another operator (unary operator)
            bool is_binary_operator = (i > 0);
            
            if (is_binary_operator) {
                char prev = expr[i-1];
                // Check if previous char is an operator or left parenthesis (making this a unary + or -)
                if (prev == '+' || prev == '-' || prev == '*' || prev == '/' || prev == '^' || prev == '(' || prev == 'e' || prev == 'E') {
                    is_binary_operator = false;  // This is likely a sign or part of scientific notation
                }
            }
            
            if (is_binary_operator) {
                if (!current.empty()) {
                    terms.push_back(current);
                    current.clear();
                }
                
                // If it's a minus sign, we add it to the next term
                if (c == '-') {
                    current += c;
                }
                // We skip the + sign as it's implied between terms
                continue;
            }
        }
        
        // Add the character to the current term
        current += c;
    }
    
    if (!current.empty()) {
        terms.push_back(current);
    }
    
    return terms;
}

template<typename T>
T monte_carlo_integrate_2d_cpu(size_t samples, T a, T b, T c, T d, const std::vector<Token>& postfix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist_x(a, b);
    std::uniform_real_distribution<T> dist_y(c, d);
    
    T sum = 0;
    size_t valid = 0;
    
    for (size_t i = 0; i < samples; ++i) {
        T x = dist_x(gen);
        T y = dist_y(gen);
        
        try {
            T value = evaluate_postfix<T>(postfix, x, y);
            if (std::isfinite(value)) {
                sum += value;
                valid++;
            }
        } catch (...) {}
    }
    
    if (valid == 0) return 0;
    T area = (b - a) * (d - c);
    return area * sum / valid;
}

int main() {
    std::cout << "Mixed Precision Monte Carlo Double Integration for Subexpressions (CUDA)\n";

    // Check CUDA device properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting...\n";
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using CUDA device: " << deviceProp.name << " with " << deviceProp.multiProcessorCount << " SMs\n";

    std::string expr;
    long double a, b, c, d;
    size_t samples;
    double tolerance = 1e-5;

    std::cout << "Function (e.g., x*y + sin(x) + cos(y)): ";
    std::getline(std::cin, expr);
    std::cout << "Lower bound for x (a): "; std::cin >> a;
    std::cout << "Upper bound for x (b): "; std::cin >> b;
    std::cout << "Lower bound for y (c): "; std::cin >> c;
    std::cout << "Upper bound for y (d): "; std::cin >> d;
    std::cout << "Number of samples: "; std::cin >> samples;
    std::cin.ignore();

    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    auto terms = split_expression(expr);

    long double total_cuda = 0;
    long double total_cpu = 0;

    std::cout << "\n--- Starting CUDA Integration ---\n";
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda);
    
    for (const auto& term : terms) {
        try {
            auto postfix = to_postfix(tokenize(term));
            long double avg = average_value(postfix, a, b, c, d);
            long double grad = estimate_gradient(postfix, a, b, c, d);
            long double var = estimate_variance(postfix, a, b, c, d, samples);

            Precision p = select_precision(avg, grad, var, tolerance);
            std::cout << "Subexpression: \"" << term << "\" | Precision: ";

            cudaEvent_t sub_start, sub_stop;
            cudaEventCreate(&sub_start);
            cudaEventCreate(&sub_stop);
            cudaEventRecord(sub_start);
            
            if (p == Precision::Float) {
                std::cout << "float";
                total_cuda += monte_carlo_integrate_2d_cuda<float>(samples, a, b, c, d, postfix);
            } else if (p == Precision::Double) {
                std::cout << "double";
                total_cuda += monte_carlo_integrate_2d_cuda<double>(samples, a, b, c, d, postfix);
            } else {
                std::cout << "long double (using double in CUDA)";
                // CUDA doesn't support long double natively, so we use double instead
                total_cuda += monte_carlo_integrate_2d_cuda<double>(samples, a, b, c, d, postfix);
            }
            
            cudaEventRecord(sub_stop);
            cudaEventSynchronize(sub_stop);
            float sub_time = 0;
            cudaEventElapsedTime(&sub_time, sub_start, sub_stop);
            std::cout << " | Time: " << (sub_time / 1000.0) << " sec\n";
            
            cudaEventDestroy(sub_start);
            cudaEventDestroy(sub_stop);

        } catch (const std::exception& e) {
            std::cerr << "Error in term \"" << term << "\": " << e.what() << "\n";
        }
    }
    
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    float cuda_time = 0;
    cudaEventElapsedTime(&cuda_time, start_cuda, stop_cuda);
    cuda_time /= 1000.0; // Convert to seconds
    
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    std::cout << "\n--- Starting CPU Integration (Single-threaded) ---\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    for (const auto& term : terms) {
        try {
            auto postfix = to_postfix(tokenize(term));
            long double avg = average_value(postfix, a, b, c, d);
            long double grad = estimate_gradient(postfix, a, b, c, d);
            long double var = estimate_variance(postfix, a, b, c, d, samples);

            Precision p = select_precision(avg, grad, var, tolerance);
            if (p == Precision::Float) {
                total_cpu += monte_carlo_integrate_2d_cpu<float>(samples, a, b, c, d, postfix); // Reduced samples for CPU
            } else if (p == Precision::Double) {
                total_cpu += monte_carlo_integrate_2d_cpu<double>(samples, a, b, c, d, postfix); // Reduced samples for CPU
            } else {
                total_cpu += monte_carlo_integrate_2d_cpu<double>(samples, a, b, c, d, postfix); // Reduced samples for CPU
            }
        } catch (...) {}
    }
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    std::cout << "\n=== Results ===\n";
    std::cout << "CUDA Result ≈ " << total_cuda << " | Time: " << cuda_time << " sec\n";
    std::cout << "CPU Result   ≈ " << total_cpu << " | Time: " << cpu_time << " sec\n";
    std::cout << "Speedup      ≈ " << (cpu_time / cuda_time) << "x\n";

    return 0;
}