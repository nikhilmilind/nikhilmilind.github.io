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
    &= \mathrm{LogSumExp} \left\lbrace \log w_i + \log f(x_i) + \log \Delta x \right\rbrace_{i=1}^N 
    \,.
\end{align}
$$

This only works when $w_i > 0$ and $f(x_i) > 0$, which happens to be the case for Boole integration
and PDFs respectively.

## Implementation

`jax` is a nice library that provides numerical gradients through functions, as long as all
operations are written using their API. This ends up being simple because `jax` provides drop-in
replacements for functions in `numpy` and `scipy`. Here is a simple implementation that sets up a
grid and integrates in log space.

```python
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
approximate the integral. However, we are often interested in taking integrals over $\mathbb{R}$.
For this purpose, consider the sigmoid transform

$$
u = \frac{\exp (k^{-1} x)}{1 + \exp (k^{-1} x)} = \mathrm{sigmoid}(k^{-1} x) \,,
$$

with inverse mapping

$$
x = k \log \left( \frac{u}{1-u} \right) = k \mathrm{logit}(u)
$$

and differential

$$
dx = \frac{k}{u(1-u)} \,du \,.
$$

Under this transformation, the interval $(-\infty, \infty)$ maps to $[0, 1]$. Furthermore,

$$
\int_{\mathbb{R}} f(x) \, dx = \int_0^1 \frac{k}{u(1-u)} f \left( k \mathrm{logit}(u) \right) \, 
du \,.
$$

In log space, with a grid of equidistant points $u_1, ..., u_N$ over $[0, 1]$,

$$
\begin{align}
    \log \int_{\mathbb{R}} f(x) \, dx &= \log \int_0^1 \frac{k}{u(1-u)} f \left( k \mathrm{logit} 
    (u) \right) \, du \\
    &\approx \mathrm{LogSumExp} \left\lbrace \log w_i + \log k - \log u_i - \log (1 - u_i) + \log f 
    \left( k \mathrm{logit} (u_i) \right) + \log \Delta u \right\rbrace_{i=1}^N \,.
\end{align}
$$

## Example

We can use a simple normal-normal model

$$
\begin{align}
    x_i \mid z_i &\sim \mathcal{N} (z_i, 1) \\
    z_i \mid \theta &\sim \mathcal{N} (\theta, 1)
\end{align}
$$

as an example. First, we need to confirm that the quadrature rule has been implemented correctly. We
can use properties of the normal distribution to get the analytical form of the evidence. First,
I write $z_i$ and $x_i$ as

$$
\begin{gather}
    z_i = \theta + \delta_i \,, \quad \delta_i \sim \mathcal{N}(0, 1) \\
    x_i = z_i + \epsilon_i \,, \quad \epsilon_i \sim \mathcal{N}(0, 1) \,.
\end{gather}
$$

Then, it becomes clear that

$$
x_i \mid \theta \sim \mathcal{N}(\theta, 2) \,.
$$

The log evidence is then

$$
\log \prod_{i=1}^n p(x_i \mid \theta) = -\sum_{i=1}^n \frac{1}{2} \log (4\pi) + \frac{1}{4}(x_i -
\theta)^2
$$

I simulated some data under this model using the following code.

```python
theta = jnp.array(5.0)

key = jax.random.key(42)
z = jax.random.normal(key, 100) + theta

use_key, key = jax.random.split(key)
x = jax.random.normal(use_key, 100) + z
```

I then estimate the log evidence using quadrature, and compare it to our analytic expectations.

```python
import jax.scipy as jsp

def log_integrand(u, x, theta, k=100):
    k_logit_u = k * (jnp.log(u) - jnp.log(1 - u))
    log_likelihood = jsp.stats.norm.logpdf(x, loc=k_logit_u, scale=1)
    log_prior = jsp.stats.norm.logpdf(k_logit_u, loc=theta, scale=1)
    return log_likelihood + log_prior + jnp.log(k) - jnp.log(u) - jnp.log(1 - u) 

integrator = BooleLogIntegrator(1001, (1E-6, 1.0 - 1E-6))
print(integrator.integrate(lambda u: log_integrand(u, x[0], theta)))

print(jsp.stats.norm.logpdf(x[0], loc=theta, scale=jnp.sqrt(2)))
```

The output is around -1.43 for both, so the quadrature is working as expected! Now, I should be able
to ascend the log evidence using numerical gradients provided by `jax`. I used the `optax` library
to perform stochastic gradient ascent on the log evidence. The resulting estimate is $\hat{\theta}
\approx 5.03$, which is pretty close to the simulated $\theta = 5$. I could test the convergence
using multiple simulations and building confidence intervals.

```python
import optax
import tqdm

def loss(theta, x):
    return -integrator.integrate(lambda u: log_integrand(u, x, theta))

loss_val_grad = jax.value_and_grad(loss)

theta_hat = jnp.array(0.0)

optimizer = optax.adam(1E-1)
opt_state = optimizer.init(theta_hat)

for i in (pbar := tqdm.trange(30)):
   for x_obs in x:
        loss_val, loss_grad = loss_val_grad(theta_hat, x_obs)
        updates, opt_state = optimizer.update(loss_grad, opt_state)
        theta_hat = optax.apply_updates(theta_hat, updates)

print(theta_hat)
```
