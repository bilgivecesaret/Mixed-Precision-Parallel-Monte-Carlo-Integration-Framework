# Mixed Precision Parallel Monte Carlo Integration Framework via OpenMP 
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;Monte Carlo integration is a well-established numerical technique for approximating definite integrals
using random sampling. The fundamental principle of Monte Carlo integration for a function f(x)
over a domain Ω is given by:</p>
<p align="center">
$\int_{\Omega} f(x) \, dx \approx V(\Omega) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1)
</p>
where V (Ω) is the volume of the domain, N is the number of random samples, and xi are random
points uniformly distributed within Ω.

&nbsp;&nbsp;&nbsp;&nbsp;The standard error of this estimate decreases proportionally to $\frac{1}{\sqrt{N}}$:
<p align="center">
$\sigma \approx \frac{\sigma_f}{\sqrt{N}}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2)
</p>
where σf is the standard deviation of the function values.
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;Several studies have explored optimizing this method through:
- Mixed precision computing: Using lower precision arithmetic (float vs double) where
appropriate to improve performance while maintaining accuracy. Works by Higham et al.
(2019) demonstrated that mixed precision strategies can significantly reduce computation time
while controlling error propagation.
- Parallel computing for Monte Carlo methods: Numerous studies have shown that MC
integration is “embarrassingly parallel” and benefits greatly from multi-threading. The work
by Lee et al. (2021) shows near-linear scaling up to hundreds of threads for certain problems.
- Hybrid algorithms: Recent work has explored combining Monte Carlo with deterministic
quadrature methods for multi-dimensional integrals, leveraging the strengths of each approach
(Giles, 2015).</p>


# Required Compiler Settings for OpenMP 
g++ -fopenmp monte_carlo_framework.cpp -o monte_carlo

# Required Compiler Settings for CUDA
nvcc monte_carlo_Cuda.cu -o monte_carlo 

# Run Command
./monte_carlo
