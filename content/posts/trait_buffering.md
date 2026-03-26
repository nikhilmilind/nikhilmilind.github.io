---
title: Trait buffering of gene dosage response curves
date: 2026-03-25
---

<!-- Trait Buffering Model and Inference (D and E, I.6) -->

In the first part of my thesis, I explored the properties of gene dosage response curves (GDRCs)
using loss-of-function (LoF) variants and duplications in the UK Biobank. Much of this work has been
described in a [pre-print](https://doi.org/10.1101/2024.11.11.24317065). This is one of multiple
posts containing some of the supplementary material that I found particularly interesting but
could not highlight in the main text.

# Introduction

We suggest two models in the pre-print that may explain why the average gene dosage response curve 
(aGDRC) is non-monotone. Curves from these models would appear to be buffered against one trait 
direction. Specifically, for traits with a negative aGDRC, the curves would appear to be buffered
against increasing trait value (and vice versa).

We created a measure to estimate this effect, which we named trait buffering. Similar to 
monotonicity, interpreting this measure requires some care, for which I have a separate post.

In the first section, we develop the actual measure itself. In the second section, we discuss the 
problem of inference using stochastic approximation expectation maximization (SAEM). In the last 
section, we look at these estimates and compare them to the aGDRCs.

# Trait Buffering Model

In the trait buffering model, we are interested in evaluating if GDRCs are systematically buffered 
against one trait direction. This could occur if non-monotone GDRCs preferentially point in one 
direction over the other. This can also occur if monotone GDRCs achieve larger values for the trait 
in one direction compared to the other.

## Distribution of Effect Sizes

We use the multivariate adaptive shrinkage (MASH) model 
[[1](https://doi.org/10.1038/s41588-018-0268-8)] as a flexible prior for our latent effect sizes. 
The MASH prior consists of a mixture of multivariate normal distributions. Suppose 
$\left\{ \mathbf{V}_k \right\}_{k=1}^K$ represents a fixed set of fixed covariance matrices in 
$\mathbb{R}^{2\times 2}$ for $K$ mixture components. We use bivariate covariance matrices over a 
grid of variances and correlation values. Let $\boldsymbol{\uppi} \in \Delta_{K-1}$ represent the 
mixture weights. Then, the gene-level prior is
$$
\boldsymbol{\upgamma}_{\cdot j} \mid \boldsymbol{\uppi} \sim \sum_{k=1}^K \pi_k\, \mathcal{N} 
\left( \mathbf{0} , \mathbf{V}_k \right) \,.
$$

The prior over the entire set of $M$ genes is
$$
\boldsymbol{\upgamma} \mid \boldsymbol{\uppi} \sim \prod_{j=1}^M \sum_{k=1}^K \pi_k \, \mathcal{N} 
\left( \boldsymbol{\upgamma}_{\cdot j} \,; \mathbf{0}, \mathbf{V}_k \right) \,.
$$

The likelihood model remains the same as before,
$$
\mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \mid \boldsymbol{\upgamma} \sim 
\mathcal{N} \left( \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \boldsymbol{\upgamma}, 
\boldsymbol{\Lambda} \right) \,.
$$

The joint likelihood is
$$
\begin{aligned}
    p \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}}, 
    \boldsymbol{\upgamma} \mid \boldsymbol{\uppi} \right) &= p \left( \mathbf{U}^\top 
    \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \mid \boldsymbol{\upgamma} \right) p 
    \left( \boldsymbol{\upgamma} \mid \boldsymbol{\uppi} \right) \\
    &= \mathcal{N} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \,; 
    \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \boldsymbol{\upgamma},
    \boldsymbol{\Lambda} \right) \prod_{j=1}^M \sum_{k=1}^K \pi_k \, \mathcal{N} \left( 
    \boldsymbol{\upgamma}_{\cdot j} \,; \mathbf{0}, \mathbf{V}_k \right) \,.
\end{aligned}
$$

## Parameter of Interest

We define the estimand of interest as
$$
\xi \stackrel{\triangle}= \frac{1}{M} \sum_{j=1}^M \log_2 \left( \frac{3}{2} \right) \gamma_{1j} + \gamma_{2j} \,.
$$

The estimand represents the dot product of $\boldsymbol{\upgamma}_{\cdot j}$ with the normal vector 
of the diagonal separating the two types of buffering within our model (Figure 1). The diagonal line 
satisfies
$$
\log_2 \left( \frac{3}{2} \right) \gamma_{1j} - \gamma_{2j} = 0 \,.
$$

Thus, GDRCs on this diagonal line are linear GDRCs (Figure 1).

<img src="/static/posts/trait_buffering/xi_hyperplane_concept.svg" class="mx-auto d-block" width="400px" />

**Figure 1:** The boundary between buffering curves is represented by the yellow gene dosage 
response curves on the left. These map to a diagonal across the burden effect plot. To estimate 
signal in one direction of the diagonal versus the other, we use the normal vector in black shown on 
the right.

Each gene in the summation for $\xi$ is either monotone or non-monotone. Genes are monotone if 
$\gamma_{1j} \gamma_{2j} < 0$ (because the signs are opposite), while genes are non-monotone if 
$\gamma_{1j} \gamma_{2j} > 0$ (because the signs are the same). We can rewrite the sum as 
$$
\xi = \frac{1}{M} \left[ \sum_{j : \gamma_{1j} \gamma_{2j} < 0} \log_2 \left( \frac{3}{2} \right) 
\gamma_{1j} + \gamma_{2j} \right] + \frac{1}{M} \left[ \sum_{j : \gamma_{1j} \gamma_{2j} > 0} 
\log_2 \left( \frac{3}{2} \right) \gamma_{1j} + \gamma_{2j} \right] \,,
$$

where the first term represents all the monotone genes, and the second term represents all the 
non-monotone genes. We define these two terms as
$$
\begin{aligned}
    \xi_{\mathrm{m}} &\stackrel{\triangle}= \frac{1}{M} \sum_{j : \gamma_{1j} \gamma_{2j} < 0} 
    \log_2 \left( \frac{3}{2} \right) \gamma_{1j} + \gamma_{2j} \\
    \xi_{\mathrm{nm}} &\stackrel{\triangle}= \frac{1}{M} \sum_{j : \gamma_{1j} \gamma_{2j} > 0} 
    \log_2 \left( \frac{3}{2} \right) \gamma_{1j} + \gamma_{2j} \,,
\end{aligned}
$$

so that
$$
\xi = \xi_{\mathrm{m}} + \xi_{\mathrm{nm}} \,.
$$

# Trait Buffering Inference

In our model, the mixture weights, $\boldsymbol{\uppi}$, are unknown. Following the method in MASH 
[[1](https://doi.org/10.1038/s41588-018-0268-8),[2](https://doi.org/10.1093/biostatistics/kxw041)], 
we use an empirical Bayes approach that combines both frequentist and Bayesian techniques. The first 
step, the frequentist step of empirical Bayes, involves maximizing the marginal likelihood to 
estimate $\hat{\boldsymbol{\uppi}}$, which we do using stochastic approximation expectation 
maximization (SAEM) \parencite{delyon_convergence_1999, kuhn_coupling_2004}. The second step, the 
Bayesian step of empirical Bayes, uses $\hat{\boldsymbol{\uppi}}$ as a plug-in estimate of 
$\boldsymbol{\uppi}$ for downstream posterior sampling.

The ideal approach to obtain the MLE of $\boldsymbol{\uppi}$ is to directly maximize the marginal 
likelihood
$$
p \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \mid \boldsymbol{\uppi} 
\right) = \int_{\mathbb{R}^{2M}} p \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
\hat{\boldsymbol{\upgamma}}, \boldsymbol{\upgamma} \mid \boldsymbol{\uppi} \right) \, 
d\boldsymbol{\upgamma} \,.
$$

However, this integral is intractable. Instead, we can use expectation maximization (EM) to maximize
this marginal likelihood \parencite{dempster_maximum_1977}. Suppose $\boldsymbol{\uppi}^{(t)}$ 
represents the estimate at the $t$th iteration of the EM algorithm. The expectation step involves 
evaluating
$$
Q \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right) \stackrel{\triangle}= 
\mathbb{E}_{\boldsymbol{\upgamma} \sim p \left( \cdot \mid \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
\hat{\boldsymbol{\upgamma}}, \boldsymbol{\uppi} = \boldsymbol{\uppi}^{(t)} \right)} \left[ \log p 
\left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}}, \boldsymbol{\upgamma} 
\mid \boldsymbol{\uppi} \right) \right] \,.
$$

Since the analytical form of $Q \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right)$ is 
also intractable under our model, we use SAEM to approximate the expectation. We sample 
$\boldsymbol{\upgamma}^{(1)}, ..., \boldsymbol{\upgamma}^{(B)} \sim p \left( \cdot \mid \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}}, \boldsymbol{\uppi} = \boldsymbol{\uppi}^{(t)} \right)$ 
using Hamiltonian Monte Carlo 
[[3](http://jmlr.org/papers/v20/18-403.html),[4](http://arxiv.org/abs/1912.11554)]. Then, we 
approximate the expectation with
$$
\mathbb{E}_{\boldsymbol{\upgamma} \sim p \left( \cdot \mid \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
\hat{\boldsymbol{\upgamma}}, \boldsymbol{\uppi} = \boldsymbol{\uppi}^{(t)} \right)} \left[ \log p
\left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}}, \boldsymbol{\upgamma} 
\mid \boldsymbol{\uppi} \right) \right] \approx \frac{1}{B} \sum_{b=1}^B \log p \left( 
\mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}}, \boldsymbol{\upgamma}^{(b)} 
\mid \boldsymbol{\uppi}=\boldsymbol{\uppi}^{(t)} \right) \,.
$$

To speed up computation, we use stochastic gradient ascent, with each chromosome representing one 
mini-batch [1](https://doi.org/10.1038/s41588-018-0268-8). In the mini-batch approach, each 
expectation is a convex sum of the previous expectation and the current expectation. Let 
$(c_t)_{t \geq 1}$ be a positive, decreasing sequence such that $\sum_{t=1}^\infty c_t = \infty$ 
and $\sum_{t=1}^\infty c_t^2 < \infty$. We use
$$
\widetilde{Q} \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right) \stackrel{\triangle}= 
\left( 1 - c_t \right) \widetilde{Q} \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t-1)} 
\right) + \frac{c_t}{B} \sum_{b = 1}^B \log p \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
\hat{\boldsymbol{\upgamma}}, \boldsymbol{\upgamma}^{(b)} \mid \boldsymbol{\uppi} = 
\boldsymbol{\uppi}^{(t)} \right)
$$

in the expectation step of EM. To define a valid recursion, we set 
$\widetilde{Q} \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(0)} \right) = 0$. Maximization, 
the second step in EM, is performed using a fixed number of iterations of stochastic gradient ascent 
with a fixed step size to obtain
$$
\boldsymbol{\uppi}^{(t+1)} = \underset{\boldsymbol{\uppi}}{\operatorname{arg max}} \widetilde{Q} 
\left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right) \,.
$$

Gradients were estimated using automatic differentiation 
[[5](https://mlsys.org/Conferences/doc/2018/146.pdf),[6](http://arxiv.org/abs/1811.05031)]. For 
each batch, we used 50 iterations of gradient descent, using AdamW to optimize
$\widetilde{Q} \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right)$ with an exponential 
decay learning rate schedule [[7](http://github.com/google-deepmind)]. We run SAEM for at most five 
epochs, and estimate $Q \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right)$ across all 
chromosomes at the end of each epoch. We stop early if the epoch estimate of 
$Q \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right)$ decreases, and set 
$\hat{\boldsymbol{\uppi}}$ to the parameter value that maximized the epoch estimate of 
$Q \left( \boldsymbol{\uppi} \mid \boldsymbol{\uppi}^{(t)} \right)$.

We use Hamiltonian Monte Carlo 
[[3](http://jmlr.org/papers/v20/18-403.html),[4](http://arxiv.org/abs/1912.11554)] to sample from 
the posterior distribution of $\boldsymbol{\upgamma}$. Following the empirical Bayes approach,
$\hat{\boldsymbol{\uppi}}$ is used as a plug-in estimate. That is, we draw samples 
$\boldsymbol{\upgamma}^{(1)}, ..., \boldsymbol{\upgamma}^{(B)} \sim p \left( \cdot \mid \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}}, \boldsymbol{\uppi} = \hat{\boldsymbol{\uppi}} \right)$. 
We use the posterior samples to estimate the cumulative density function of the posterior 
distribution of $\xi$. Specifically, if
$$
\xi^{(b)} = \frac{1}{M} \sum_{j=1}^M \log_2 \left( \frac{3}{2} \right) \gamma_{1j}^{(b)} + 
\gamma_{2j}^{(b)}
$$

represents the value of $\xi$ for the $b$th sample, then the empirical cumulative density function 
for the posterior distribution is
$$
\hat{\mathbb{P}} \left( \xi \leq x \,\Big|\, \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
\hat{\boldsymbol{\upgamma}} \right) = \frac{1}{B} \sum_{b=1}^B \mathbb{I} \left[ \xi^{(b)} 
\leq x \right] \,,
$$

where $\mathbb{I} \left[ \cdot \right]$ is the indicator function. Under this definition, the 
posterior mean is estimated as
$$
\mathbb{E} \left[ \xi \,\Big|\, \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} 
\right] \approx \frac{1}{B} \sum_{b=1}^B \xi^{(b)} \,.
$$

Furthermore, we estimated the 95% credible interval by using the 2.5th and the 97.5th percentile of 
the posterior density. The same procedure is used for $\xi_{\mathrm{m}}$ and $\xi_{\mathrm{nm}}$.

# Trait Buffering Estimates

![](/static/posts/trait_buffering/xi.svg)

**Figure 2:** (Left) The posterior estimates of trait buffering, $\xi$, decomposed into 
contributions from non-monotone and monotone GDRCs. These estimates are derived from genes with a 
local false sign rate of less than 10% for the LoF and duplication burden effects. (Right) The 
estimates of the components remains non-zero with increasing confidence in the signs of the 
gene-level effects.

$\xi$ is the average signed deviation of GDRCs from linearity, and should be concordant with 
the aGDRCs of the trait. For example, a trait experiencing negative trait buffering will have GDRCs 
that decrease a trait more than they increase the trait (Figure 5A). The aGDRC for these curves 
should be in the negative direction, which implies that the average LoF burden effect and the 
average duplication burden effect are negative. Indeed, we observe that our estimates of $\xi$ align 
well with the aGDRCs of the traits (Figure 3).

<img src="/static/posts/trait_buffering/xi_agdrc.svg" class="mx-auto d-block" width="400px" />

**Figure 3:** The average LoF and duplication burden effects for the traits in our analysis. The 
data presented here is the same as Figure 1B. Large points represents traits with a nominally 
significant average LoF burden effect and average duplication burden effect at the $\alpha = 0.05$ 
level. The color of the points represents the sign of the estimate of $\xi$ for all genes with a 
local false sign rate of less than 5%.

