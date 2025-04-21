#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <limits.h>


double function(double x) {
    return sin(x);
}


// Monte Carlo Integration 
double monte_carlo(float a, float b, long long n, int thread_count) {
    float sum = 0.0;
    #pragma omp parallel num_threads(thread_count) reduction(+:sum)
    {
        int seed = rand() * INT_MAX ;
        float local_sum = 0.0f;
        #pragma omp for
        for (long long i = 0; i < n; i++) {
            float x = a + ((float)rand_r(&seed) / RAND_MAX) * (b - a);
            local_sum += function(x);
        }
        sum += (double)local_sum;
    }
    return (b - a) * sum / n;
}


int main() {
    double a = 0.0, b = M_PI; // integral limits
    long long n = 100000000;  //  sample numbers

    printf("Monte Carlo Integration: sin(x) from %.2f to %.2f with %lld samples\n", a, b, n);

    // Time Measurement (paralel)
    double start_time = omp_get_wtime();
    double result_parallel = monte_carlo(a, b, n, omp_get_max_threads());
    double parallel_elapsed_time = omp_get_wtime() - start_time;

    // Time Measurement (serial)
    start_time = omp_get_wtime();
    double result_serial = monte_carlo(a, b, n, 1);
    double serial_elapsed_time = omp_get_wtime() - start_time;


    // Actual Value
    double exact = 2.0;

    // Results
    double speedup = serial_elapsed_time/parallel_elapsed_time;
    double effiency = speedup / omp_get_max_threads();
    printf("Exact value = %.15f\n", exact);
    printf("Result of Serial   = %.15f | Error = %.15f | Serial Time   = %.7f sec\n", result_serial, fabs(result_serial - exact), serial_elapsed_time);
    printf("Result of Parallel = %.15f | Error = %.15f | Parallel Time = %.7f sec\n", result_parallel, fabs(result_parallel - exact), parallel_elapsed_time);    
    printf("Speedup  = %.5fX\n", speedup);
    printf("Effiency = %.2f %%\n",effiency);

    return 0;
}