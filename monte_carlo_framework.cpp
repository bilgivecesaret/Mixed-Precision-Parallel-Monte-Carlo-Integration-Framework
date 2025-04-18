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

enum class TokenType { Number, Operator, Function, Variable, LeftParen, RightParen };

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

// Tokenizer
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
        } else if (isalpha(c)) {
            std::string name;
            while (i < expr.length() && isalpha(expr[i])) name += expr[i++];
            if (name == "x") tokens.emplace_back(TokenType::Variable, name);
            else tokens.emplace_back(TokenType::Function, name);
        } else if (std::string("+-*/^").find(c) != std::string::npos) {
            tokens.emplace_back(TokenType::Operator, std::string(1, c));
            ++i;
        } else if (c == '(') {
            tokens.emplace_back(TokenType::LeftParen, "(");
            ++i;
        } else if (c == ')') {
            tokens.emplace_back(TokenType::RightParen, ")");
            ++i;
        } else {
            throw std::runtime_error(std::string("Invalid character: ") + c);
        }
    }
    return tokens;
}

// Precedence and associativity
int precedence(const std::string& op) {
    if (op == "+" || op == "-") return 1;
    if (op == "*" || op == "/") return 2;
    if (op == "^") return 3;
    return 0;
}
bool is_right_associative(const std::string& op) {
    return op == "^";
}

// Infix to postfix (Shunting Yard Algorithm)
std::vector<Token> to_postfix(const std::vector<Token>& tokens) {
    std::vector<Token> output;
    std::stack<Token> stack;

    for (const auto& token : tokens) {
        if (token.type == TokenType::Number || token.type == TokenType::Variable) {
            output.push_back(token);
        } else if (token.type == TokenType::Function) {
            stack.push(token);
        } else if (token.type == TokenType::Operator) {
            while (!stack.empty() && (
                (stack.top().type == TokenType::Function) ||
                (stack.top().type == TokenType::Operator &&
                ((precedence(stack.top().value) > precedence(token.value)) ||
                (precedence(stack.top().value) == precedence(token.value) &&
                 !is_right_associative(token.value))))) ) {
                output.push_back(stack.top());
                stack.pop();
            }
            stack.push(token);
        } else if (token.type == TokenType::LeftParen) {
            stack.push(token);
        } else if (token.type == TokenType::RightParen) {
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

// Postfix evaluator (templated)
template<typename T>
T evaluate_postfix(const std::vector<Token>& postfix, T x_val) {
    std::stack<T> stack;
    for (const auto& token : postfix) {
        if (token.type == TokenType::Number) {
            stack.push(static_cast<T>(std::stold(token.value)));
        } else if (token.type == TokenType::Variable) {
            stack.push(x_val);
        } else if (token.type == TokenType::Operator) {
            T b = stack.top(); stack.pop();
            T a = stack.top(); stack.pop();
            if (token.value == "+") stack.push(a + b);
            else if (token.value == "-") stack.push(a - b);
            else if (token.value == "*") stack.push(a * b);
            else if (token.value == "/") stack.push(a / b);
            else if (token.value == "^") stack.push(std::pow(a, b));
        } else if (token.type == TokenType::Function) {
            T a = stack.top(); stack.pop();
            if (token.value == "sin") stack.push(std::sin(a));
            else if (token.value == "cos") stack.push(std::cos(a));
            else if (token.value == "log") stack.push(std::log10(a));
            else if (token.value == "ln") stack.push(std::log(a));
            else if (token.value == "exp") stack.push(std::exp(a));
            else if (token.value == "sqrt") stack.push(std::sqrt(a));
            else throw std::runtime_error("Unknown function: " + token.value);
        }
    }
    return stack.top();
}

// Precision types
enum class Precision { Float, Double, LongDouble };

// Heuristic precision selector
Precision select_precision(long double avg_val, long double grad, long double var, double tolerance) {
    if (var > 1e3 || grad > 1e2) return Precision::LongDouble;
    if (avg_val > 1e3 || var > 1e1 || grad > 1e1) return Precision::Double;
    if (tolerance < 1e-5) return Precision::Float;
    return Precision::Float;
}

// Monte Carlo integration for templated type
template<typename T>
T monte_carlo_integrate(size_t samples, T a, T b, const std::vector<Token>& postfix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(static_cast<double>(a), static_cast<double>(b));
    T sum = 0;

    #pragma omp parallel
    {
        T local_sum = 0;
        #pragma omp for
        for (size_t i = 0; i < samples; ++i) {
            T x = static_cast<T>(dist(gen));
            T val = evaluate_postfix<T>(postfix, x);
            local_sum += val;
        }
        #pragma omp critical
        sum += local_sum;
    }

    return (b - a) * sum / samples;
}

// Average value over interval
long double average_value(const std::vector<Token>& postfix, long double a, long double b) {
    long double sum = 0;
    for (int i = 0; i < 100; ++i) {
        long double x = a + (b - a) * i / 99.0;
        sum += std::abs(evaluate_postfix<long double>(postfix, x));
    }
    return sum / 100.0;
}

// Variance estimator
long double estimate_variance(const std::vector<Token>& postfix, long double a, long double b, size_t samples) {
    long double sum = 0, sum_sq = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<long double> dist(a, b);

    for (size_t i = 0; i < samples; ++i) {
        long double x = dist(gen);
        long double val = evaluate_postfix<long double>(postfix, x);
        sum += val;
        sum_sq += val * val;
    }
    long double mean = sum / samples;
    return (sum_sq / samples) - (mean * mean);
}

// Gradient approximation
long double estimate_gradient(const std::vector<Token>& postfix, long double a, long double b) {
    long double delta = (b - a) / 100.0;
    return std::abs(evaluate_postfix<long double>(postfix, b) - evaluate_postfix<long double>(postfix, a)) / delta;
}

int main() {
    std::cout << "Mixed Precision Monte Carlo Integration for Subexpressions\n";

    std::string expr;
    long double a, b;
    size_t samples;
    double tolerance = 1e-4;

    std::cout << "Function (e.g., x^2 + 2*x + 1): ";
    std::getline(std::cin, expr);    
    std::cout << "Upper bound b: "; std::cin >> b;
    std::cout << "Lower bound a: "; std::cin >> a;
    std::cout << "Number of samples: "; std::cin >> samples;
    std::cin.ignore();

    std::stringstream ss(expr);
    std::string term;
    long double total = 0.0;

    while (std::getline(ss, term, '+')) {
        auto tokens = tokenize(term);
        auto postfix = to_postfix(tokens);
        long double avg = average_value(postfix, a, b);
        long double grad = estimate_gradient(postfix, a, b);
        long double var = estimate_variance(postfix, a, b, samples);

        Precision p = select_precision(avg, grad, var, tolerance);
        std::cout << "Subexpression: \"" << term << "\" | Precision: ";

        if (p == Precision::Float) {
            std::cout << "float\n";
            total += monte_carlo_integrate<float>(samples, (float)a, (float)b, postfix);
        } else if (p == Precision::Double) {
            std::cout << "double\n";
            total += monte_carlo_integrate<double>(samples, (double)a, (double)b, postfix);
        } else {
            std::cout << "long double\n";
            total += monte_carlo_integrate<long double>(samples, a, b, postfix);
        }
    }

    std::cout << "\nTotal integral result ≈ " << total << "\n";
    return 0;
}
