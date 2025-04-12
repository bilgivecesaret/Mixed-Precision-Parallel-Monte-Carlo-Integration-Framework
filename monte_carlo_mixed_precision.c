#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// The function we will integrate is
double function(double x) {
    return sin(x);
}

// Float version
float functionf(float x) {
    return sinf(x);
}

// Monte Carlo Integration (float)
float monte_carlo_float(float a, float b, long long n) {
    float sum = 0.0f;
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        float local_sum = 0.0f;
        #pragma omp for
        for (long long i = 0; i < n; i++) {
            float x = a + ((float)rand_r(&seed) / RAND_MAX) * (b - a);
            local_sum += functionf(x);
        }
        #pragma omp atomic
        sum += local_sum;
    }
    return (b - a) * sum / n;
}

// Monte Carlo Integration (double)
double monte_carlo_double(double a, double b, long long n) {
    double sum = 0.0;
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        double local_sum = 0.0;
        #pragma omp for
        for (long long i = 0; i < n; i++) {
            double x = a + ((double)rand_r(&seed) / RAND_MAX) * (b - a);
            local_sum += function(x);
        }
        #pragma omp atomic
        sum += local_sum;
    }
    return (b - a) * sum / n;
}

int main() {
    double a = 0.0, b = M_PI; // integral limits
    long long n = 100000000;  //  sample numbers

    printf("Monte Carlo Integration: sin(x) from %.2f to %.2f with %lld samples\n", a, b, n);

    // Time Measurement (double)
    double start_time = omp_get_wtime();
    double result_double = monte_carlo_double(a, b, n);
    double time_double = omp_get_wtime() - start_time;

    // Time Measurement (float)
    start_time = omp_get_wtime();
    float result_float = monte_carlo_float((float)a, (float)b, n);
    double time_float = omp_get_wtime() - start_time;

    // Actual Value
    double exact = 2.0;

    // Results
    printf("\n[DOUBLE]  Result = %.8f | Error = %.8f | Time = %.4f sec\n", result_double, fabs(result_double - exact), time_double);
    printf("[FLOAT ]  Result = %.8f | Error = %.8f | Time = %.4f sec\n", result_float, fabs(result_float - exact), time_float);

    return 0;
}
