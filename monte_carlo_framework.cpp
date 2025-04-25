#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cmath>
#include <cctype>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h>
#include <sstream>
#include <mutex>

enum class TokenType { Number, Operator, Function, Variable, LeftParen, RightParen };

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

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
    max_val = std::max(max_val, 1.0L);  // Don't let very small values ​​distort the error estimate.
    grad = std::max(grad, 1.0L);    // If the gradient is almost zero, the minimum error estimate can still be made.


    // Estimate error as epsilon * max gradient * scale
    long double error_float = eps_float * max_val * grad;
    long double error_double = eps_double * max_val * grad;
    long double error_long_double = eps_long_double * max_val * grad;

    // Compare estimated relative error with tolerance
    if (error_float <= tol) return Precision::Float;
    if (error_double <= tol) return Precision::Double;
    return Precision::LongDouble;
}

template<typename T>
T monte_carlo_integrate_2d(size_t samples, T a, T b, T c, T d, const std::vector<Token>& postfix) {
    T sum = 0;
    size_t valid_samples = 0;
    #pragma omp parallel default(none) shared(samples, a, b, c, d, postfix, sum, valid_samples)
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<T> dist_x(a, b);
        std::uniform_real_distribution<T> dist_y(c, d);
        T local_sum = 0;
        size_t local_valid = 0;

        #pragma omp for
        for (size_t i = 0; i < samples; ++i) {
            T x = dist_x(gen);
            T y = dist_y(gen);
            try {
                local_sum += evaluate_postfix<T>(postfix, x, y);
                local_valid++;
            } catch (...) {}
        }

        #pragma omp atomic
        sum += local_sum;
        #pragma omp atomic
        valid_samples += local_valid;
    }
    if (valid_samples == 0) return 0;
    T area = (b - a) * (d - c);
    return area * sum / valid_samples; // Use valid_samples instead of samples
}

long double average_value(const std::vector<Token>& postfix, long double a, long double b, long double c, long double d) {
    const int N = 10;
    long double sum = 0;
    int valid = 0;

    #pragma omp parallel for reduction(+:sum,valid) collapse(2)
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

    #pragma omp parallel default(none) shared(postfix, a, b, c, d, samples) reduction(+:sum, sum_sq, valid)
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<long double> dist_x(a, b);
        std::uniform_real_distribution<long double> dist_y(c, d);

        #pragma omp for
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
    }
    if (valid < 2) return 0;
    long double mean = sum / valid;
    return (sum_sq / valid) - (mean * mean);
}

long double estimate_gradient(const std::vector<Token>& postfix, long double a, long double b, long double c, long double d) {
    const int N = 10;
    long double max_grad = 0;

    #pragma omp parallel for reduction(max:max_grad) collapse(2)
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

int main() {
    std::cout << "Mixed Precision Monte Carlo Double Integration for Subexpressions\n";

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

    long double total_parallel = 0;
    long double total_serial = 0;

    omp_set_num_threads(6);
    std::cout << "\n--- Starting Parallel Integration ["<<omp_get_max_threads()<<"]---\n";
    double start_parallel = omp_get_wtime();
    for (const auto& term : terms) {
        try {
            auto postfix = to_postfix(tokenize(term));
            long double avg = average_value(postfix, a, b, c, d);
            long double grad = estimate_gradient(postfix, a, b, c, d);
            long double var = estimate_variance(postfix, a, b, c, d, samples);

            Precision p = select_precision(avg, grad, var, tolerance);
            std::cout << "Subexpression: \"" << term << "\" | Precision: ";

            double sub_start = omp_get_wtime();
            if (p == Precision::Float) {
                std::cout << "float";
                total_parallel += monte_carlo_integrate_2d<float>(samples, a, b, c, d, postfix);
            } else if (p == Precision::Double) {
                std::cout << "double";
                total_parallel += monte_carlo_integrate_2d<double>(samples, a, b, c, d, postfix);
            } else {
                std::cout << "long double";
                total_parallel += monte_carlo_integrate_2d<long double>(samples, a, b, c, d, postfix);
            }
            double sub_end = omp_get_wtime();
            std::cout << " | Time: " << (sub_end - sub_start) << " sec\n";

        } catch (const std::exception& e) {
            std::cerr << "Error in term \"" << term << "\": " << e.what() << "\n";
        }
    }
    double end_parallel = omp_get_wtime();
    double parallel_time = end_parallel - start_parallel;

    std::cout << "\n--- Starting Serial Integration (1 thread) ---\n";
    omp_set_num_threads(1);
    double start_serial = omp_get_wtime();
    for (const auto& term : terms) {
        try {
            auto postfix = to_postfix(tokenize(term));
            long double avg = average_value(postfix, a, b, c, d);
            long double grad = estimate_gradient(postfix, a, b, c, d);
            long double var = estimate_variance(postfix, a, b, c, d, samples);

            Precision p = select_precision(avg, grad, var, tolerance);
            if (p == Precision::Float) {
                total_serial += monte_carlo_integrate_2d<float>(samples, a, b, c, d, postfix);
            } else if (p == Precision::Double) {
                total_serial += monte_carlo_integrate_2d<double>(samples, a, b, c, d, postfix);
            } else {
                total_serial += monte_carlo_integrate_2d<long double>(samples, a, b, c, d, postfix);
            }
        } catch (...) {}
    }
    double end_serial = omp_get_wtime();
    double serial_time = end_serial - start_serial;

    std::cout << "\n=== Results ===\n";
    std::cout << "Parallel Result ≈ " << total_parallel << " | Time: " << parallel_time << " sec\n";
    std::cout << "Serial Result   ≈ " << total_serial << " | Time: " << serial_time << " sec\n";
    std::cout << "Speedup         ≈ " << (serial_time / parallel_time) << "x\n";

    return 0;
}