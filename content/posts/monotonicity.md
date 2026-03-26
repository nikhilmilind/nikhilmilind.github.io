---
title: Measuring the monotonicity of gene dosage response curves
date: 2026-03-24
---

In the first part of my thesis, I explored the properties of gene dosage response curves (GDRCs)
using loss-of-function (LoF) variants and duplications in the UK Biobank. Much of this work has been
described in a [pre-print](https://doi.org/10.1101/2024.11.11.24317065). This is one of multiple
posts containing some of the supplementary material that I found particularly interesting but
could not highlight in the main text.

# Introduction

Once we realized that we could not ascertain GDRCs using the summary statistics at
individual genes, we decided to measure monotonicity across genes for a single trait. Interpreting
this measure requires some care, for which I have a separate post.

In the first section, we develop the actual measure itself and discuss the distribution of some of 
the random components in the model. In the second section, we discuss the problem of inference using
either maximum likelihood or method-of-moments. In the last section, we compare estimates from the
two methods.

# Monotonicity

Here, we endeavor to jointly model burden summary statistics from different points on the dosage 
spectrum. Burden summary statistics are often reported as an effect size and a standard error from 
regression. We use hierarchical models to account for the sampling error and to pool information 
across genes.

Let $\mathbf{Y} \in \mathbb{R}^N$ be a standardized phenotype of interest, measured in $N$ 
individuals. Let $\mathbf{X}, \mathbf{Z} \in \mathbb{R}^{N \times M}$ represent the burden genotype 
matrices for two variant classes respectively. By variant class, we mean a set of variants with a 
dosage effect in the same direction, such as deletions, duplications, or LoF variants. We assume 
that there are $M$ genes that are polymorphic for both classes of variants. The genotypes are 
encoded as copies of the dosage-perturbing allele.

We use the following linear system to model the phenotype:
$$
\mathbf{Y} = \mathbf{X} \boldsymbol{\upgamma}_1 + \mathbf{Z} \boldsymbol{\upgamma}_2 + 
\boldsymbol{\upepsilon} \,.
$$

Here, $\boldsymbol{\upgamma}_1, \boldsymbol{\upgamma}_2 \in \mathbb{R}^M$ are the unobserved, 
per-allele effect sizes of perturbations of each gene on the phenotype. The residual error is 
represented by $\boldsymbol{\upepsilon} \in \mathbb{R}^N$ and is assumed to be drawn from an 
isotropic normal distribution,
$$
\boldsymbol{\upepsilon} \sim \mathcal{N} \left( \mathbf{0}, \sigma_\epsilon^2 \mathbf{I} \right) \,.
$$

## Distribution of Effect Sizes

We model the latent effect sizes as coming from an uncentered normal distribution. Effect sizes
between genes are assumed to be independent, but effect sizes for the same gene covary between the
variant classes. That is to say, the effect of a deletion or LoF variant is assumed to carry some 
information about the effect of a duplication of the same gene. This covariance structure is 
represented with diagonal matrices $\boldsymbol{\Sigma}_{11}$, $\boldsymbol{\Sigma}_{12}$, and
$\boldsymbol{\Sigma}_{22}$:
$$
\begin{bmatrix}
    \boldsymbol{\upgamma}_1\\
    \boldsymbol{\upgamma}_2
\end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix}
    \overline{\boldsymbol{\upgamma}}_1\\
    \overline{\boldsymbol{\upgamma}}_2
\end{bmatrix}, \begin{bmatrix}
    \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12}\\
    \boldsymbol{\Sigma}_{12}^\top & \boldsymbol{\Sigma}_{22}
\end{bmatrix} \right) \,.
$$

Suppose we collect the block matrices above into 
$\boldsymbol{\upgamma}, \overline{\boldsymbol{\upgamma}} \in \mathbb{R}^{2M}$ and 
$\boldsymbol{\Sigma} \in \mathbb{R}^{2M \times 2M}$. Then, we can succinctly state that
$$
\boldsymbol{\upgamma} \sim \mathcal{N} \left( \overline{\boldsymbol{\upgamma}}, \boldsymbol{\Sigma}
\right) \,.
$$

Let $\boldsymbol{\upgamma}_{\cdot j} \in \mathbb{R}^2$ represent the latent effect sizes for the 
$j$th gene. For instance, the first coordinate may represent the LoF effect, and the second
coordinate may represent the duplication effect. Then, the marginal distribution is written as
$$
\begin{bmatrix}
    \gamma_{1j}\\
    \gamma_{2j}
\end{bmatrix} \overset{\mathrm{iid}}{\sim} \mathcal{N} \left( \begin{bmatrix}
    \overline{\gamma}_1\\
    \overline{\gamma}_2
\end{bmatrix}, \begin{bmatrix}
    \sigma_{11}^2 & \sigma_{12}\\
    \sigma_{12} & \sigma_{22}^2
\end{bmatrix} \right) \,.
$$

## Parameter of Interest

We consider a gene to have a "monotone" effect on a trait if the dosage-reducing alleles have an
opposite direction of effect compared to dosage-increasing alleles. To this end, we define the 
monotonicity, $\phi$, as a value proportional to the negative uncentered second moment of the effect
sizes,
$$
\phi \stackrel{\triangle}\propto -\mathbb{E} \left[ \gamma_{1j} \gamma_{2j} \right] \,.
$$

Thus, a positive $\phi$ represents monotone behavior. To compare monotonicity across traits, we
restrict its codomain to $[-1, 1]$. We do this using the Cauchy-Schwarz inequality, which guarantees
that
$$
\left\lvert \mathbb{E} \left[ \gamma_{1j} \gamma_{2j} \right] \right\rvert^2 \leq \mathbb{E} \left[
\gamma_{1j}^2 \right] \mathbb{E} \left[ \gamma_{2j}^2 \right] \,,
$$

so that
$$
\phi \stackrel{\triangle}= -\frac{ \mathbb{E} \left[ \gamma_{1j} \gamma_{2j} \right] }{ \sqrt{ 
\mathbb{E} \left[ \gamma_{1j}^2 \right] \mathbb{E} \left[ \gamma_{2j}^2 \right] } } = -\frac{ 
\overline{\gamma}_{1} \overline{\gamma}_{2} + \sigma_{12} }{ \sqrt{ \left( \overline{\gamma}_{1}^2 + 
\sigma_{11}^2 \right) \left( \overline{\gamma}_{2}^2 + \sigma_{22}^2 \right) } } \,.
$$

If the effect sizes have mean zero (that is, $\overline{\gamma}_{1} = \overline{\gamma}_{2} = 0$),
this is equivalent to the negative correlation coefficient,
$$
\phi = -\rho = -\frac{ \sigma_{12} }{ \sigma_{11} \sigma_{22} } \,.
$$

## Distribution of Genotypes

We consider burden genotypes in this analysis, which represent an aggregate of the genotypes across
multiple sites. Specifically, we collapse different alleles that have the same effect on dosage into
a single allele for each gene. Burden genotypes are encoded using the same scheme as biallelic
variants.

We begin by assuming that there is no correlation between the different variant classes. This is
reasonable to assume because different variants arise through different mutational processes, and
burden genotypes represent aggregates of multiple rare variants, each of which generally have 
negligible linkage disequilibrium (LD) [[1](https://doi.org/10.1093/genetics/iyac004)]. In our data,
for the vast majority of genes (97%), there is no overlap between individuals that carry LoF
variants and individuals that carry duplications. For 290 genes (3%), we observe at least one
individual with both a LoF variant and duplication. Even within these 290 genes, overlap of
individuals with LoF variants and individuals with duplications is rare &mdash; on average, less 
than 1% of LoF carriers also have a duplication in these 290 genes. However, this assumption does 
not apply across genes within a variant class. Within duplications, for instance, large duplications
consisting of multiple genes will induce correlations between the burden genotypes. This burden
genotype correlation is similar to LD between biallelic genotypes at unique sites.

We model the different variant classes at a gene separately. We assume that the $M$ burden genotypes
are in Hardy-Weinberg equilibrium (HWE), which implies a binomial likelihood. Let $p_j$ represent
the allele frequency of the $j$th gene's dosage-perturbing allele for the first variant class and
let $q_j$ represent the allele frequency of the $j$th gene's dosage-perturbing allele for the second
variant class. Without loss of generality, we assume that the genotypes are centered but not scaled:
$$
\begin{aligned}
    \mathbf{X}_{ij} + 2p_j &\sim \mathrm{Binomial} \left( 2, p_j \right) \\
    \mathbf{Z}_{ij} + 2q_j &\sim \mathrm{Binomial} \left( 2, q_j \right) \,.
\end{aligned}
$$

Our assumptions imply that there is no correlation between variant classes,
$$
\mathbb{E} \left[ \mathbf{X}_{ij} \mathbf{Z}_{k\ell} \right] = 0 \quad \forall i, j, k, \ell \,.
$$

However, we do model the correlation between the same variant class. The correlation matrices
$\mathbf{R}_1$ and $\mathbf{R}_2$ are used to represent this correlation. Let 
$P_j = 2p_j \left( 1 - p_j \right)$ and $Q_j = 2q_j \left( 1 - q_j \right)$ represent the 
heterozygosity of the burden genotypes for the $j$th gene. Then,
$$
\begin{aligned}
    \mathbb{E} \left[ \mathbf{X}^\top \mathbf{X} \right] &= \mathbf{P}^{\frac{1}{2}} \mathbf{R}_1 
    \mathbf{P}^{\frac{1}{2}}\\
    \mathbb{E} \left[ \mathbf{Z}^\top \mathbf{Z} \right] &= \mathbf{Q}^{\frac{1}{2}} \mathbf{R}_2 
    \mathbf{Q}^{\frac{1}{2}} \,,
\end{aligned}
$$

where
$$
\begin{aligned}
    \mathbf{P} &= \mathrm{diag} \left( \left\{ P_j \right\}_{j=1}^M \right)\\
    \mathbf{Q} &= \mathrm{diag} \left( \left\{ Q_j \right\}_{j=1}^M \right) \,.
\end{aligned}
$$

Because LoF variants affect individual genes, and we restrict ourselves to analyzing rare variants,
we assume that the LoF burden genotypes are independent and that there is no correlation between
them. That is, $\mathbf{R}_1 = \mathbf{I}$.

## Burden Regression

Burden test results are reported as summary statistics from regression between the burden genotype
and the phenotype. Regression is performed on centered and scaled phenotypes. The marginal summary 
statistics are calculated based on the following model for each gene:
$$
\begin{aligned}
    \mathbf{Y} &= \mathbf{X}_{\cdot j} \gamma_{1j} + \boldsymbol{\upepsilon}_1\\
    \mathbf{Y} &= \mathbf{Z}_{\cdot j} \gamma_{2j} + \boldsymbol{\upepsilon}_2 \,.
\end{aligned}
$$

The effect size is approximately the following. Note that
$\left( \mathbf{X}_{\cdot j}^\top \mathbf{X}_{\cdot j} \right)^{-1} \approx \frac{1}{N P_j}$ and 
$\left( \mathbf{Z}_{\cdot j}^\top \mathbf{Z}_{\cdot j} \right)^{-1} \approx \frac{1}{N Q_j}$, so
$$
\begin{alignat*}{2}
    \hat{\gamma}_{1j} &= \left( \mathbf{X}_{\cdot j}^\top \mathbf{X}_{\cdot j} \right)^{-1} 
    \mathbf{X}_{\cdot j}^\top \mathbf{Y} &\approx \frac{1}{N P_j} \mathbf{X}_{\cdot j}^\top 
    \mathbf{Y}\\
    \hat{\gamma}_{2j} &= \left( \mathbf{Z}_{\cdot j}^\top \mathbf{Z}_{\cdot j} \right)^{-1} 
    \mathbf{Z}_{\cdot j}^\top \mathbf{Y} &\approx \frac{1}{N Q_j} \mathbf{Z}_{\cdot j}^\top 
    \mathbf{Y} \,.
\end{alignat*}
$$

The standard error is approximately the following. Here, we assume that any individual gene explains 
a small fraction of the total variance. That is, $\sigma_{\epsilon_1}^2 \approx 1$ and 
$\sigma_{\epsilon_2}^2 \approx 1$.  Thus,
$$
\begin{alignat*}{3}
    \hat{s}_{1j} &\stackrel{\triangle}= \widehat{\mathrm{SE}} \left( \hat{\gamma}_{1j} \right) &= 
    \sqrt{\sigma_{\epsilon_1}^2 \left( \mathbf{X}_{\cdot j}^\top \mathbf{X}_{\cdot j} \right)^{-1} } 
    &\approx \frac{1}{\sqrt{N P_j}}\\
    \hat{s}_{2j} &\stackrel{\triangle}= \widehat{\mathrm{SE}} \left( \hat{\gamma}_{2j} \right) &= 
    \sqrt{\sigma_{\epsilon_2}^2 \left( \mathbf{Z}_{\cdot j}^\top \mathbf{Z}_{\cdot j} \right)^{-1} } 
    &\approx \frac{1}{\sqrt{N Q_j}} \,.
\end{alignat*}
$$

Therefore, the Z scores are
$$
\begin{aligned}
    z_{1j} &\approx \frac{1}{\sqrt{N P_j}} \mathbf{X}_{\cdot j}^\top \mathbf{Y}\\
    z_{2j} &\approx \frac{1}{\sqrt{N Q_j}} \mathbf{Z}_{\cdot j}^\top \mathbf{Y} \,.
\end{aligned}
$$

# Monotonicity Inference

## Maximum Likelihood Estimation

For inference, we will use the summary statistics directly rather than the underlying genotype data.
That is, we are given the estimated effect sizes ($\hat{\boldsymbol{\upgamma}}_1$ and
$\hat{\boldsymbol{\upgamma}}_2$) and their standard errors ($\hat{\mathbf{s}}_1$ and
$\hat{\mathbf{s}}_2$). To simplify notation, we represent the prior as
$$
\boldsymbol{\upgamma} \sim \mathcal{N} \left( \overline{\boldsymbol{\upgamma}} \left(
\boldsymbol{\uptheta} \right), \boldsymbol{\Sigma} \left( \boldsymbol{\upomega} \right) \right) \,,
$$

where $\boldsymbol{\uptheta}$ and $\boldsymbol{\upomega}$ represent values used to parameterize the
priors.

Next, the likelihood of the estimated effect sizes is defined using the approximate regression with
summary statistics (RSS) likelihood [[2](https://doi.org/10.1214/17-aoas1046)]. Under this
likelihood, the estimated effect sizes are patterned by the burden genotype correlation in the
cohort. Furthermore, the standard error of the sampling distribution is used as an estimate of the
dispersion around the mean. [Zhu and Stephens]((https://doi.org/10.1214/17-aoas1046)) showed that
this likelihood asymptotically approaches the sampling distribution of the estimated effect sizes.
Let $\hat{\mathbf{S}}_1 = \mathrm{diag} \left( \left\{ s_{1j} \right\}_{j=1}^M \right)$ and 
$\hat{\mathbf{S}}_2 = \mathrm{diag} \left( \left\{ s_{2j} \right\}_{j=1}^M \right)$, and let 
$\hat{\mathbf{R}}_1$ and $\hat{\mathbf{R}}_2$ represent the in-sample correlation matrices of the 
two variant classes respectively. Then, in notation reflecting the RSS likelihood, we define
$$
\begin{aligned}
    \hat{\mathbf{S}} &\stackrel{\triangle}= \begin{bmatrix}
        \hat{\mathbf{S}}_1 & \mathbf{0}\\
        \mathbf{0} & \hat{\mathbf{S}}_2
    \end{bmatrix}\\
    \hat{\mathbf{R}} &\stackrel{\triangle}= \begin{bmatrix}
        \hat{\mathbf{R}}_1 & \mathbf{0}\\
        \mathbf{0} & \hat{\mathbf{R}}_2
    \end{bmatrix} \,.
\end{aligned}
$$

Then, the RSS likelihood for the observed effect sizes is
$$
\hat{\boldsymbol{\upgamma}} \mid \boldsymbol{\upgamma} \sim \mathcal{N} \left( \hat{\mathbf{S}}
\hat{\mathbf{R}} \hat{\mathbf{S}}^{-1} \boldsymbol{\upgamma}, \hat{\mathbf{S}} \hat{\mathbf{R}}
\hat{\mathbf{S}} \right) \,.
$$

Some genes are perfectly correlated with each other. That is, the sample correlation matrix 
$\hat{\mathbf{R}}$ is not strictly positive definite. To improve numerical stability and to account 
for perfect correlation, we project the data onto the linear subspace of dimension $L < M$ spanned
by the correlation matrix (that is, a projection orthogonal to the null space). Consider the 
following eigendecomposition, with a matrix with orthogonal columns of eigenvectors 
$\mathbf{U} \in \mathbb{R}^{M \times L}$ and a diagonal matrix of positive eigenvalues 
$\boldsymbol{\Lambda} \in \mathbb{R}^{L\times L}$,
$$
\hat{\mathbf{R}} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top \,.
$$

In practice, these are derived by dropping small eigenvalues from the numerical eigendecomposition 
of $\hat{\mathbf{R}}$. Rather than modeling the observations $\hat{\boldsymbol{\upgamma}}$, we model 
the projected data,
$$
\mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \mid \boldsymbol{\upgamma} \sim 
\mathcal{N} \left( \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \boldsymbol{\upgamma}, 
\boldsymbol{\Lambda} \right) \,.
$$

Note that under this model, the marginal likelihood of the estimates is
$$
\mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \sim \mathcal{N} \left( \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \overline{\boldsymbol{\upgamma}}, \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \boldsymbol{\Sigma} \hat{\mathbf{S}}^{-1} \mathbf{U} \boldsymbol{\Lambda} + \boldsymbol{\Lambda} \right) \,.
$$

To simplify notation, let $\mathbf{A} = \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1}$ 
and let $\mathbf{K} = \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^\top + \boldsymbol{\Lambda}$. Then,
$$
\mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \sim \mathcal{N} \left( \mathbf{A} 
\overline{\boldsymbol{\upgamma}}, \mathbf{K} \right) \,.
$$

The pseudoinverse $\mathbf{A}^+$ is a useful quantity in downstream derivations. We can construct a 
pseudoinverse using components of the eigendecomposition. The pseudoinverse is
$$
\mathbf{A}^+ = \hat{\mathbf{S}} \mathbf{U} \boldsymbol{\Lambda}^{-1} \,.
$$

This pseudoinverse is a right pseudoinverse, as can be seen by
$$
\begin{aligned}
    \mathbf{A} \mathbf{A}^+ &= \boldsymbol{\Lambda} \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
    \hat{\mathbf{S}} \mathbf{U} \boldsymbol{\Lambda}^{-1} \\
    &= \boldsymbol{\Lambda} \mathbf{U}^\top \mathbf{U} \boldsymbol{\Lambda}^{-1} \\
    &= \boldsymbol{\Lambda} \boldsymbol{\Lambda}^{-1} \\
    &= \mathbf{I} \,.
\end{aligned}
$$

The other properties of the pseudoinverse are readily confirmed using matrix algebra.

### Inference

We maximize the marginal likelihood of the observed effect sizes with respect to the parameters of
the model. The likelihood is
$$
\begin{aligned}
    \mathcal{L} \left( \boldsymbol{\uptheta}, \boldsymbol{\upomega} \mid 
    \hat{\boldsymbol{\upgamma}}, \hat{\mathbf{s}} \right) &= \mathcal{N} \left( \mathbf{U}^\top 
    \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} \,; \mathbf{A} 
    \overline{\boldsymbol{\upgamma}}, \mathbf{K} \right)\\
    &= \left( 2\pi \right)^{-M} \left( \det \mathbf{K} \right)^{-\frac{1}{2}} \exp \left( 
    -\frac{1}{2} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - 
    \mathbf{A} \overline{\boldsymbol{\upgamma}} \right)^\top \mathbf{K}^{-1} \left( 
    \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} 
    \overline{\boldsymbol{\upgamma}} \right) \right) \,.
\end{aligned}
$$

The log likelihood is
$$
\begin{aligned}
    \ell \left( \boldsymbol{\uptheta}, \boldsymbol{\upomega} \mid \hat{\boldsymbol{\upgamma}}, 
    \hat{\mathbf{s}} \right) &= \log \mathcal{L} \left( \boldsymbol{\upgamma}, \boldsymbol{\upomega} 
    \mid \hat{\boldsymbol{\upgamma}}, \hat{\mathbf{s}} \right)\\
    &= -M \log \left( 2\pi \right) - \frac{1}{2} \log \det \mathbf{K} - \frac{1}{2} \left( 
    \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} 
    \overline{\boldsymbol{\upgamma}} \right)^\top \mathbf{K}^{-1} \left( \mathbf{U}^\top 
    \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} 
    \overline{\boldsymbol{\upgamma}} \right) \,.
\end{aligned}
$$

We are interested in obtaining the maximum likelihood estimates $\hat{\boldsymbol{\uptheta}}$ and 
$\hat{\boldsymbol{\upomega}}$ such that
$$
\hat{\boldsymbol{\uptheta}}, \hat{\boldsymbol{\upomega}} = \underset{\boldsymbol{\uptheta}, 
\boldsymbol{\upomega}}{\operatorname{arg max}} \mathcal{L} \left( \boldsymbol{\uptheta}, 
\boldsymbol{\upomega} \mid \hat{\boldsymbol{\upgamma}}, \hat{\boldsymbol{s}} \right) \,.
$$

For maximum likelihood estimation of $\boldsymbol{\uptheta}$ and $\boldsymbol{\upgamma}$, we use 
natural gradient ascent \parencite{amari_natural_1998}. This is an optimization approach that uses 
both first- and second-order information about the log likelihood via the gradient and Fisher 
information matrix. The gradient of the log likelihood function with respect to 
$\overline{\boldsymbol{\gamma}}$ is
$$
\begin{aligned}
    \nabla_{\overline{\boldsymbol{\upgamma}}} \, \ell &= \nabla_{\overline{\boldsymbol{\upgamma}}} \, \left[  - \frac{1}{2} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} \overline{\boldsymbol{\upgamma}} \right)^\top \mathbf{K}^{-1} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} \overline{\boldsymbol{\upgamma}} \right) \right]\\
    &= \mathbf{A}^\top \mathbf{K}^{-1} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} \overline{\boldsymbol{\upgamma}} \right) \,.
\end{aligned}
$$

Using the matrix chain rule, the derivative of the log likelihood with respect to one of the mean 
parameters is
$$
\frac{\partial \ell}{\partial \theta_i} = \mathrm{Tr} \left[ \left( 
\nabla_{\overline{\boldsymbol{\upgamma}}} \, \ell \right)^\top \frac{\partial 
\overline{\boldsymbol{\upgamma}}}{\partial \theta_i} \right] = \left( 
\nabla_{\overline{\boldsymbol{\upgamma}}} \, \ell \right)^\top \frac{\partial 
\overline{\boldsymbol{\upgamma}}}{\partial \theta_i} \,.
$$

Therefore, it follows that the gradient of the log likelihood with respect to the mean parameters is
$$
\nabla_{\boldsymbol{\uptheta}} \, \ell = \left( \nabla_{\overline{\boldsymbol{\upgamma}}} \, \ell 
\right)^\top \frac{\partial \overline{\boldsymbol{\upgamma}}}{\partial \boldsymbol{\uptheta}} \,.
$$

The matrix derivative of the log likelihood with respect to $\mathbf{K}$ is
$$
\begin{aligned}
    \frac{\partial \ell}{\partial \mathbf{K}} &= \frac{\partial}{\partial \mathbf{K}} \left[ - 
    \frac{1}{2} \log \det \mathbf{K} - \frac{1}{2} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} 
    \hat{\boldsymbol{\upgamma}} - \mathbf{A} \overline{\boldsymbol{\upgamma}} \right)^\top 
    \mathbf{K}^{-1} \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - 
    \mathbf{A} \overline{\boldsymbol{\upgamma}} \right) \right]\\
    &= -\frac{1}{2} \mathbf{K}^{-1} + \frac{1}{2} \mathbf{K}^{-1} \left( \mathbf{U}^\top 
    \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} \overline{\boldsymbol{\upgamma}} 
    \right) \left( \mathbf{U}^\top \hat{\mathbf{S}}^{-1} \hat{\boldsymbol{\upgamma}} - \mathbf{A} 
    \overline{\boldsymbol{\upgamma}} \right)^\top \mathbf{K}^{-1} \,.
\end{aligned}
$$

The derivative of $\mathbf{K}$ with respect to a given covariance parameter is
$$
\frac{\partial \mathbf{K}}{\partial \omega_i} = \mathbf{A} \frac{\partial 
\boldsymbol{\Sigma}}{\partial \omega_i} \mathbf{A}^\top \,.
$$

Using the matrix chain rule, the derivative with respect to a given covariance parameter is
$$
\frac{\partial \ell}{\partial \omega_i} = \mathrm{Tr} \left[ \left( \frac{\partial \ell}{\partial 
\mathbf{K}} \right)^\top \frac{\partial \mathbf{K}}{\partial \omega_i} \right] \,.
$$

The Fisher information for multivariate normal distributions has a special form 
[[3](https://doi.org/10.1145/2725494.2725510)] such that the mean and covariance parameters do not 
share any information,
$$
\boldsymbol{\mathcal{I}} \left( \boldsymbol{\uptheta}, \boldsymbol{\upomega} \right) = 
\begin{bmatrix}
    \boldsymbol{\mathcal{I}}_{\overline{\boldsymbol{\upgamma}}} \left( \boldsymbol{\uptheta}, 
    \boldsymbol{\upomega} \right) & \mathbf{0}\\
    \mathbf{0} & \boldsymbol{\mathcal{I}}_{\mathbf{K}} \left( \boldsymbol{\upomega} \right)
\end{bmatrix} \,.
$$

The Fisher information of the mean parameter is
$$
\left[ \boldsymbol{\mathcal{I}}_{\overline{\boldsymbol{\upgamma}}} \left( \boldsymbol{\uptheta}, 
\boldsymbol{\upomega} \right) \right]_{ij} = \left( \frac{\partial 
\overline{\boldsymbol{\upgamma}}}{\partial \theta_i} \Bigg|_{\boldsymbol{\uptheta}} \right)^\top 
\mathbf{A}^\top \left[ \mathbf{K} \left( \boldsymbol{\upomega} \right) \right]^{-1} \mathbf{A} 
\left( \frac{\partial \overline{\boldsymbol{\upgamma}}}{\partial \theta_j} 
\Bigg|_{\boldsymbol{\uptheta}} \right) \,.
$$

The Fisher information of the covariance parameters is
$$
\left[ \boldsymbol{\mathcal{I}}_{\mathbf{K}} \left( \boldsymbol{\upomega} \right) \right]_{ij} = 
\frac{1}{2} \mathrm{Tr} \left[ \left[ \mathbf{K} \left( \boldsymbol{\upomega} \right) \right]^{-1} 
\left( \frac{\partial \mathbf{K}}{\partial \omega_i} \Bigg|_{\boldsymbol{\upomega}} \right) \left[ 
\mathbf{K} \left( \boldsymbol{\upomega} \right) \right]^{-1} \left( \frac{\partial 
\mathbf{K}}{\partial \omega_j} \Bigg|_{\boldsymbol{\upomega}} \right) \right] \,.
$$

Natural gradient ascent involves Newton-Raphson updates with the Fisher information matrix. A 
dampening parameter $0 < \alpha_t \leq 1$ is chosen using a backtracking line search
[[4](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-16/issue-1/Minimization-of-functions-having-Lipschitz-continuous-first-partial-derivatives/pjm/1102995080.full)] 
at each iteration to improve stability and serve as a stopping condition,
$$
\begin{aligned}
    \hat{\boldsymbol{\uptheta}}_{t + 1} &= \hat{\boldsymbol{\uptheta}}_t + \alpha_t \left[ 
    \boldsymbol{\mathcal{I}}_{\overline{\boldsymbol{\upgamma}}} \left( 
    \hat{\boldsymbol{\uptheta}}_t, \hat{\boldsymbol{\upomega}}_t \right) \right]^{-1} \left[ 
    \nabla_{\boldsymbol{\uptheta}} \, \ell \,\Big|_{\boldsymbol{\uptheta} = 
    \hat{\boldsymbol{\uptheta}}_t} \right]\\
    \hat{\boldsymbol{\upomega}}_{t + 1} &= \hat{\boldsymbol{\upomega}}_t + \alpha_t \left[ 
    \boldsymbol{\mathcal{I}}_{\mathbf{K}} \left( \hat{\boldsymbol{\upomega}}_t \right) \right]^{-1} 
    \left[ \nabla_{\boldsymbol{\upomega}} \, \ell \,\Big|_{\boldsymbol{\upomega} = 
    \hat{\boldsymbol{\upomega}}_t} \right] \,.
\end{aligned}
$$

In practice, we estimate the gradients and Fisher information matrix for each chromosome separately 
and sum them up because effect estimates are assumed to be independent across chromosomes. The 
derivatives 
$\frac{\partial \overline{\boldsymbol{\upgamma}}}{\partial \theta_i}$ and 
$\frac{\partial \mathbf{K}}{\partial \omega_i}$ are computed using automatic differentiation 
[[5](https://mlsys.org/Conferences/doc/2018/146.pdf),[6](http://arxiv.org/abs/1811.05031)].

### Uncertainty Estimation

We use the delta method to estimate the uncertainty in $\hat{\phi}$. Since $\phi \in [-1, 1]$, the 
delta method struggles to estimate the uncertainty near the boundaries. Instead, we estimate the 
standard error for
$$
\phi' = \frac{1}{2} \log \left( \frac{1 + \phi}{1 - \phi} \right) \,,
$$

which is the Z transformation used to estimate confidence intervals for correlation coefficients. 
The inverse map is
$$
\phi = \frac{\exp \left( 2 \phi' \right) - 1}{\exp \left( 2\phi' \right) + 1} \,.
$$


Let $\hat{\boldsymbol{\uptheta}}$ and $\hat{\boldsymbol{\upomega}}$ be the maximum likelihood 
estimate (MLE) of $\boldsymbol{\uptheta}$ and $\boldsymbol{\upomega}$ respectively. By the 
convergence properties of the MLE,
$$
\begin{aligned}
    \hat{\boldsymbol{\uptheta}} &\overset{\mathrm{d}}{\longrightarrow} \mathcal{N} \left( 
    \boldsymbol{\uptheta}, \left[ 
    \boldsymbol{\mathcal{I}}_{\overline{\boldsymbol{\upgamma}}} \left( \boldsymbol{\uptheta}, 
    \boldsymbol{\upomega} \right) \right]^{-1} \right)\\
    \hat{\boldsymbol{\upomega}} &\overset{\mathrm{d}}{\longrightarrow} \mathcal{N} \left( 
    \boldsymbol{\upomega}, \left[ 
    \boldsymbol{\mathcal{I}}_{\mathbf{K}} \left( \boldsymbol{\upomega} \right) \right]^{-1}
    \right) \,.
\end{aligned}
$$

By the delta method, the estimator for $\phi'$ converges to 
$$
\hat{\phi}' \overset{\mathrm{d}}{\longrightarrow} \mathcal{N} \left( \phi', \left[ 
\nabla_{\boldsymbol{\uptheta}} \phi' 
\Big|_{\boldsymbol{\uptheta}, \boldsymbol{\upomega}} \right]^\top \left[ 
\boldsymbol{\mathcal{I}}_{\overline{\boldsymbol{\upgamma}}} \left( \boldsymbol{\uptheta}, 
\boldsymbol{\upomega} \right) \right]^{-1} \left[ \nabla_{\boldsymbol{\uptheta}} \phi' 
\Big|_{\boldsymbol{\uptheta}, \boldsymbol{\upomega}} \right] + \left[ \nabla_{\boldsymbol{\upomega}} 
\phi' \Big|_{\boldsymbol{\uptheta}, \boldsymbol{\upomega}} \right]^\top \left[ 
\boldsymbol{\mathcal{I}}_{\mathbf{K}} \left( \boldsymbol{\upomega} \right) \right]^{-1} \left[
\nabla_{\boldsymbol{\upomega}} \phi' \Big|_{\boldsymbol{\uptheta}, \boldsymbol{\upomega}} \right] 
\right) \,,
$$
which can be used to derive approximate confidence intervals. The confidence intervals for 
$\hat{\phi}'$ were inferred using this method, and the inverse map was used to determine the 
boundaries for the confidence intervals for $\hat{\phi}$.

# Monotonicity Estimates

Figures 1 and 2 show the monotonicity estimates using MLE and MoM respectively. We used various
continuous traits in the UK Biobank, which we detail in the pre-print. We found the MLE estimates to
be more precise than the MoM estimates. In general, most of the traits had a positive monotonicity.

![](/static/posts/monotonicity/phi_estimates.svg)

**Figure 1:** Point estimates with 95% confidence intervals for monotonicity from the maximum 
likelihood estimation approach. Orange points represent significantly positive values of 
$\hat{\phi}$ at the $\alpha = 0.05$ level. Gray points represent non-significant values.

![](/static/posts/monotonicity/phi_estimates_mom.svg)

**Figure 2:** Point estimates with 95% confidence intervals for monotonicity from the 
method-of-moments (MoM) approach. Orange points represent significantly positive values of 
$\hat{\phi}$ at the $\alpha = 0.05$ level. Gray points represent non-significant values.

## Comparing MLE and MoM Estimates

Our two estimators for $\phi$ are based on different approaches (maximum likelihood estimation and 
MoM), and are each guaranteed to be consistent but not be unbiased in a finite sample. For this 
section, suppose that $\hat{\phi}_{\mathrm{MLE}}$ represents the MLE and $\hat{\phi}_{\mathrm{MoM}}$ 
represents the MoM estimate. For the MLE, recall that we estimate the standard error for 
$\hat{\phi}'_{\mathrm{MLE}}$ (Section \ref*{apx:mle-uncertainty-estimation}), which is a 
transformation of $\phi$ into an unconstrained space. For the MoM estimate, we use the bootstrap to 
estimate the standard error.

We use an errors-in-variables approach to estimate the concordance between the two estimators for 
$\phi$. Suppose that
$$
\hat{\phi}'_{\mathrm{MLE}} \sim \mathcal{N} \left( \frac{1}{2} \log \left( \frac{1 + 
\phi_{\mathrm{MLE}}}{1 - \phi_{\mathrm{MLE}}} \right), \widehat{\mathrm{SE}}^2 \left( 
    \hat{\phi}'_{\mathrm{MLE}} \right) \right) \,,
$$

which is reasonable since the sampling distribution of the MLE asymptotically approaches this 
distribution. We make the stronger modeling assumption that
$$
\hat{\phi}_{\mathrm{MoM}} \sim \mathcal{N} \left( \phi_{\mathrm{MoM}}, \widehat{\mathrm{SE}}^2 
\left( \hat{\phi}_{\mathrm{MoM}} \right) \right) \,.
$$

Finally, to estimate the concordance, we assume a linear relationship,
$$
\phi_{\mathrm{MoM}} = \beta \phi_{\mathrm{MLE}} \,.
$$

We fit this non-linear, errors-in-variables model using the orthogonal distance regression routines 
implemented in `scipy` [[7](https://doi.org/10.1038/s41592-019-0686-2)]. The estimates from the two
methods were broadly concordant, with a confidently positive estimate for the regression coefficient 
(Figure 3).

<img src="/static/posts/monotonicity/phi_hat_mle_mom_comparison.svg" class="mx-auto d-block" width="400px" />

**Figure 3:** Comparing maximum likelihood estimation versus method-of-moments estimation for 
$\phi$. The regression line is estimated using total least squares (TLS). A 95% confidence interval 
for the regression estimate is included. Each point represents one of the traits selected for 
analysis
