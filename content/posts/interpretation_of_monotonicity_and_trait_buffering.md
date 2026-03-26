---
title: Interpreting monotonicity and trait buffering
date: 2026-03-27
---

In the first part of my thesis, I explored the properties of gene dosage response curves (GDRCs)
using loss-of-function (LoF) variants and duplications in the UK Biobank. Much of this work has been
described in a [pre-print](https://doi.org/10.1101/2024.11.11.24317065). This is one of multiple
posts containing some of the supplementary material that I found particularly interesting but
could not highlight in the main text.

# Introduction

In our pre-print, we introduced monotonicity and trait buffering as measures of interest. I have
discussed these at length in previous posts. Here, I describe how to interpret these measures.

# Interpretation of Monotonicity

We use $\phi$ as a measure of overall monotone signal for a trait. In Figure 1, we provide some 
examples to aid in the interpretation of $\phi$. If LoF and duplication variants have opposite 
effects on a trait for all genes, $\phi$ will be positive (Figure 1A). In contrast, if LoF variants 
and duplications in all genes have the same direction of effect on a trait, $\phi$ will be negative 
(Figure 1B). Thus, the sign of $\phi$ generally allows us to interpret the average behavior of the 
direction of effects of burden tests for a trait. The value of $\phi$ does not need to be 1 for all 
genes to be monotone, and it does not need to be -1 for all genes to be non-monotone.

![](/static/posts/interpretation_of_monotonicity_and_trait_buffering/phi_interpretation.svg)

**Figure 1:** **A.** If loss-of-function variants and duplications have opposite effects on a trait
for all genes, $\phi$ has some positive value. **B.** If loss-of-function variants and duplications 
have effects in the same direction on a trait for all genes, $\phi$ has some negative value. **C.** 
$\phi$ does not measure the number of monotone or non-monotone GDRCs. In this example, all quadrants 
have the same number of genes, but $\phi$ is positive because the monotone genes have larger effects 
on average. **D.** $\phi$ can be dominated by genes with large effects. In this example, all but one 
gene are non-monotone, but $\phi$ is still positive.

$\phi$ does not count the number of GDRCs present for a trait. As an example, consider Figure 1C, 
where each quadrant contains the same number of genes, summing to 20 monotone GDRCs and 20 
non-monotone GDRCs for the hypothetical trait. On average, the monotone GDRCs have larger effects on 
the trait, and thus $\phi$ is positive. This means that even one gene with a large effect can 
influence $\phi$ (Figure 1D).

# Interpretation of Trait Buffering

We use $\xi$ to better understanding trait buffering. The goal is to use $\xi$ to explain why the 
average GDRC is non-montone. In Figure 2A, we display the burden effect sizes of the genes for a 
hypothetical trait experiencing negative trait buffering. Negative trait buffering means that GDRCs 
are buffered against increasing the trait value. For this trait, a majority of the contribution to 
this signal comes from non-monotone GDRCs ($\xi_{\mathrm{nm}} \approx -0.23$ and 
$\xi_{\mathrm{m}} \approx -0.12$). In our main figure in the pre-print, we report the posterior 
means for the components $\xi_{\mathrm{m}}$ and $\xi_{\mathrm{nm}}$, which add up to exactly equal 
our estimate for $\xi$ based on the law of total expectation.

![](/static/posts/interpretation_of_monotonicity_and_trait_buffering/xi_interpretation.svg)

**Figure 2:** **A.** Trait buffering measures the signed deviation of GDRCs from the diagonal line, 
which explains the non-zero average GDRC (larger yellow point). **B.** Trait buffering can occur with 
only monotone GDRCs present (the average GDRC is still non-monotone). **C.** Trait buffering can 
occur with only non-monotone GDRCs present.

To understand the meaning of the contributions, we provide two examples where all of the 
contributions to trait buffering come from either monotone GDRCs (Figure 2B) or non-monotone GDRCs 
(Figure 2C).

# Real Data

We can use the posterior estimates from our trait buffering model to visualize these latent spaces 
for traits that we study. In our main figure in the pre-print, we looked specifically at body mass 
index (BMI), height, and peak expiratory flow (PEF). We can explore these using the posterior 
samples from our trait buffering model. The posterior mean effect sizes from our model for BMI, 
height, and PEF are displayed in Figures 3, 4, and 5 respectively. In each case, we can see the 
effects of trait buffering by visualizing the distance genes deviate from the diagonal line.

![](/static/posts/interpretation_of_monotonicity_and_trait_buffering/phi_and_xi_bmi.svg)

**Figure 3:** Posterior mean effect estimates for LoF and duplication burden tests from our trait 
buffering model for body mass index (BMI). The top row displays all genes, while the bottom row 
displays genes where both LoF and duplication burden tests have a local false sign rate (LFSR) of 
less than 10%. The left column displays the contribution of each gene to $\phi$, while the right 
column displays the contribution of each gene to $\xi$.

![](/static/posts/interpretation_of_monotonicity_and_trait_buffering/phi_and_xi_height.svg)

**Figure 4:** Posterior mean effect estimates for LoF and duplication burden tests from our trait 
buffering model for height. The top row displays all genes, while the bottom row displays genes 
where both LoF and duplication burden tests have a local false sign rate (LFSR) of less than 10%. 
The left column displays the contribution of each gene to $\phi$, while the right column displays 
the contribution of each gene to $\xi$.

![](/static/posts/interpretation_of_monotonicity_and_trait_buffering/phi_and_xi_pef.svg)

**Figure 5:** Posterior mean effect estimates for LoF and duplication burden tests from our trait 
buffering model for peak expiratory flow (PEF). The top row displays all genes, while the bottom row 
displays genes where both LoF and duplication burden tests have a local false sign rate (LFSR) of 
less than 10%. The left column displays the contribution of each gene to $\phi$, while the right 
column displays the contribution of each gene to $\xi$.

The contribution to $\phi$ that is used in the visualization of Figures 3, 4, and 5 is derived from 
the posterior samples for each gene. It represents the average value of 
$-\gamma_{\mathrm{LoF}} \times \gamma_{\mathrm{Dup}}$ for each gene across the posterior samples, 
and is not strictly equivalent to the contribution to $\phi$ under the monotonicity model. In 
contrast, the contribution to $\xi$ used in the visualization can be interpreted as the gene-level 
contribution to the statistic reported in our main analysis since it is based on the posterior 
samples.

BMI is a trait with large and positive $\hat{\phi}$, height is a trait with moderate and positive 
$\hat{\phi}$, and PEF is a trait with non-significant $\hat{\phi}$. These visualized latent spaces 
should thus span the various types of traits present in our data set. BMI has a positive average 
GDRC, while both height and PEF have negative average GDRCs. As expected, the direction of trait 
buffering for these traits is consistent with their average GDRCs and is visually discernible in the 
visualized latent spaces.
