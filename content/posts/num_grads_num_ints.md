---
title: Numerical gradients through numerical integrals
date: 2025-03-15
---

I recently encountered a classic situation that occurs often when playing around with empirical
Bayes. I had a hierarchical model of the following type:

$$
\begin{align}
    x \mid z &\sim F(z) \\
    z \mid \theta &\sim G(\theta) \,.
\end{align}
$$

Here, $x$ is the observed data and $\theta$ is the parameter we wish to estimate. In empirical
Bayes, we maximize the log evidence,

$$
\hat{\theta} = \arg\max_{\theta} J(\theta) = \arg\max_{\theta} \log \int_{\mathbb{R}} F(x \,; z) \,
G(z \,; \theta) \, dz \,.
$$

The problem is that this integral is often analytically intractable, and we wish to evaluate it 
numerically. However, to maximize $J(\theta)$, we also need access to the gradient 
$\nabla_\theta J$. For example, if I am using gradient ascent on the log likelihood, I will need
numerical gradients.

When dealing with probability distributions, we often evaluate the log probability density function
(PDF) instead of the PDF for numerical stability. Therefore, I wanted to implement a numerical 
integration scheme that works directly on log PDF values, which allows for numerical gradients.

## Numerical Integration

There are many different quadrature rules that aim to approximate integrals as finite sums of
evaluation points of the function. I used 
<a href="https://en.wikipedia.org/wiki/Boole%27s_rule" target="_blank">Boole's rule</a>. Like many
other quadrature rules, we can write it as

$$
\int_a^b f(x) \, dx \approx \sum_{i=1}^N w_i f(x_i) \Delta x \,,
$$

where $x_1, ..., x_N$ is a grid of $N$ equally-spaced evaluation points on $[a,b]$ for the function
and $w_1, ..., w_N$ is the set of weights for the quadrature. If we are evaluating the log integral,

$$
\begin{align}
    \log \int_a^b f(x) \, dx &\approx \log \left( \sum_{i=1}^N w_i f(x_i) \Delta x \right) \\
    &= \log \sum_{i=1}^N \exp \left[ \log \left( w_i f(x_i) \Delta x \right) \right] \\
    &= \mathrm{LogSumExp} \left\lbrace \log w_i + \log f(x_i) + \log \Delta x \right\rbrace_{i=1}^N \,.
\end{align}
$$

This only works when $w_i > 0$ and $f(x_i) > 0$, which happens to be the case for Boole integration
and PDFs respectively.

## Implementation

`jax` is a nice library that provides numerical gradients through functions, as long as all
operations are written using their API. This ends up being simple because `jax` provides drop-in
replacements for functions in `numpy` and `scipy`. Here is a simple implementation that sets up a
grid and integrates in log space.

```{python}
import jax
import jax.numpy as jnp

class BooleLogIntegrator:

    def __init__(self, N, int_domain):

        '''
        :param N: The number of integration points.
        :param int_domain: The domain of integration.
        '''
    
        self.N = N
        self.int_start, self.int_end = int_domain

        # Choose the closest N such that (N mod 4) = 1
        # Required grid for the Boole method
        self.N = (self.N // 4) * 4 + 1

        # Create a grid of equidistant points
        self.grid = jnp.linspace(self.int_start, self.int_end, self.N)
        self.delta_x = (self.int_end - self.int_start) / self.N

        # Create grids for various integration points
        self.grid_endpoints = jnp.array([0, self.N - 1])
        self.grid_1 = jnp.arange(1, self.N, 2)
        self.grid_2 = jnp.arange(2, self.N - 1, 4)
        self.grid_3 = jnp.arange(4, self.N - 3, 4)

        # Create factors for various grids
        self.grid_factor = jnp.empty(self.N)
        self.grid_factor = self.grid_factor.at[self.grid_endpoints].set(jnp.log(7.0))
        self.grid_factor = self.grid_factor.at[self.grid_1].set(jnp.log(32.0))
        self.grid_factor = self.grid_factor.at[self.grid_2].set(jnp.log(12.0))
        self.grid_factor = self.grid_factor.at[self.grid_3].set(jnp.log(14.0))
        self.grid_factor = self.grid_factor + jnp.log(2.0) + jnp.log(self.delta_x) - jnp.log(45.0)
        
    def integrate(self, f):

        '''
        Integrates a function with respect to an evaluation grid.

        :param f: The log integrand.
        '''

        evals = f(self.grid)
        evals += self.grid_factor
        return jax.nn.logsumexp(evals)
```

## Post Script on Integration Bounds

Numerical quadrature methods are built for finite intervals $[a, b]$ because we use finite sums to
approximate the integral. However, we are often interested in taking integrals over $\mathbb{R}$. For this purpose, consider the sigmoid transform

$$
u = \frac{\exp (x)}{1 + \exp (x)} = \mathrm{sigmoid}(x) \,,
$$

with inverse mapping

$$
x = \log \left( \frac{u}{1-u} \right) = \mathrm{logit}(u)
$$

and differential

$$
dx = \frac{1}{u(1-u)} \,du \,.
$$

Under this transformation, the interval $(-\infty, \infty)$ maps to $[-1, 1]$. Furthermore,

$$
\int_{\mathbb{R}} f(x) \, dx = \int_{-1}^1 \frac{1}{u(1-u)} f \left( \mathrm{logit}(u) \right) \, du \,.
$$

In log space, with a grid of equidistant points $u_1, ..., u_N$ over $[-1, 1]$,

$$
\begin{align}
    \log \int_{\mathbb{R}} f(x) \, dx &= \log \int_{-1}^1 \frac{1}{u(1-u)} f \left( \mathrm{logit} (u) \right) \, du \\
    &\approx \mathrm{LogSumExp} \left\lbrace \log w_i - \log u_i - \log (1 - u_i) + \log f \left( \mathrm{logit} (u_i) \right) + \log \Delta u \right\rbrace_{i=1}^N \,.
\end{align}
$$
