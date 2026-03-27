---
title: Assessing burden test inflation with synonymous variants and improving 
date: 2026-03-24
---

In the first part of my thesis, I explored the properties of gene dosage response curves (GDRCs)
using loss-of-function (LoF) variants and duplications in the UK Biobank. Much of this work has been
described in a [pre-print](https://doi.org/10.1101/2024.11.11.24317065). This is one of multiple
posts containing some of the supplementary material that I found particularly interesting but
could not highlight in the main text.

# Introduction

We used the whole-exome sequencing data in the UK Biobank to perform burden tests. In the pre-print,
we used loss-of-function (LoF) variants, which should reduce gene dosage by 50%. However, we wanted
to optimize the burden tests to include as many individuals in the UKB as possible. We used
synonymous variants to ensure that our results were not inflated due to population structure.

Separately, we also attempted to increase the signal in our burden tests by reducing the number of
misannotated LoF variants. LoF variants are annotaetd using computational methods, which can result
in false positives. We use a probability of misannotation that was estimated in previous work to
build masks for our burden tests.

# Synonymous Variants

Stratification or other confounding can inflate the estimated effect sizes from association 
analyses. We can ensure that we are adequately controlling for confounding by estimating the effect 
of synonymous variants, which are expected to not affect protein function and downstream traits.

We used synonymous variants to test the effect of using different subsets of individuals. The 
original analysis of LoF variants in the UK Biobank (UKB) 
[[Backman et al.](https://doi.org/10.1038/s41586-021-04103-z)] used a subset of around 430K individuals with 
genetic similarity to the EUR superpopulation from the 1000 Genomes Project. In addition to a 
population with genetic similarity to EUR (called EUR for brevity), we tested the subset of 390K 
unrelated individuals in the EUR subset (called unrelated) and the subset of all 460K individuals 
with whole-exome sequencing data (called WES). This allowed us to determine if relatedness or 
population stratification inflated our estimates of effect size. The EUR population was defined 
using self-reported information and boundaries in genotyping principal component (PC) space from 
prior genetic analysis in the UKB [[1](https://doi.org/10.1038/s41588-020-00757-z)]: 
$-20 \leq \mathrm{PC1} \leq 40$ and $-25 \leq \mathrm{PC2} \leq 10$ (Array items 1 and 2 from field 
22009 in the UKB) for either self-identified "White British" or self-identified "non-British White" 
(Field 21000 in the UKB).

Additionally, we also used synonymous variants to test the effect of using different numbers of 
genotyping PCs. The original analysis used 10 genotyping PCs when performing association analyses 
within populations with high genetic similarity [[2]((https://doi.org/10.1038/s41586-021-04103-z))].
In addition to 10 genotyping PCs, we tested 15 and 20 genotyping PCs since we planned to use as many 
individuals in the UKB as possible, which might introduce additional confounding due to population 
stratification.

To test for stratification, we ran synonymous variant burden tests on a subset of nine continuous 
traits: height, body mass index (BMI), low density lipoprotein (LDL), mean corpuscular hemoglobin 
(MCH), red blood cell distribution width (RDW), forced vital capacity (FVC), creatinine, cystatin C, 
and the north coordinate of the place of birth in the United Kingdom (NC). Strong effects for NC 
should be a good measure of uncontrolled stratification.

## Number of Genotyping Principal Components

Synonymous variants are expected to not have any effect on traits. Therefore, the mean squared 
effect of synonymous variant burden associations should provide an estimate of inflation in effect 
sizes due to other sources of confounding. The gold standard for genetic association analysis is 
using the cohort of unrelated individuals with high genetic similarity. Compared to this cohort, the 
amount of inflation was indistinguishable in the EUR and WES cohorts (Figure 1). Thus, we decided to 
use the WES cohort to maximize our sample size.

<img src="/static/posts/synonymous_variants/synonymous_variants_stratification.svg" class="mx-auto d-block" width="400px" />

**Figure 1:** The mean squared effect of synonymous variants for various cohorts of individuals and 
different numbers of genotyping PCs. The unrelated and EUR cohort burden tests used 15 genotyping 
PCs. We tested 10, 15, and 20 genotyping PCs in the WES cohort burden tests.

The effect of including 10 genotyping PCs was also indistinguishable from 15 or 20 genotyping PCs 
(Figure 1). The original analysis included 10 genotyping PCs 
[[2]((https://doi.org/10.1038/s41586-021-04103-z))], but performed analyses in cohorts of high 
genetic similarity. Since we were including all individuals in the UKB, we decided to conservatively 
use 15 genotyping PCs.

## Inflation from Confounding

Since we detected a significant mean squared effect for NC, we were concerned about inflation of 
effect sizes due to confounding. To test the effect of this inflation, we compared the mean squared 
effect of LoF variant burden associations with the mean squared effect of synonymous variant burden 
associations across various traits. We noted that the mean squared effect of LoF burden associations 
was an order of magnitude larger than the effect of synonymous burden associations (Figure 2). Thus, 
we concluded that although confounding is likely present in our association tests, the signal is at 
least 10 times greater than the bias.

<img src="/static/posts/synonymous_variants/synonymous_versus_LoFs.svg" class="mx-auto d-block" width="400px" />

**Figure 2:** LoF variants have a magnitude larger effect on traits than synonymous variants across 
various continuous traits.

## Utility of Misannotation Probability

![](/static/posts/synonymous_variants/misannotation_probability.svg)

**Figure 3:** The mean squared effect of genes in various buckets of 
$\log_{10} \left( s_{\mathrm{het}} \right)$ for various traits. Genes with larger effects tend to be 
more constrained. All data here uses a MAF < 1% filter other than the MAF < 0.001% associations from
[[Backman et al.](https://doi.org/10.1038/s41586-021-04103-z)]. The use of the misannotation 
probability increases the signal detected. The signal in the original analysis of the LoF data is 
also shown for reference. 95% confidence intervals are displayed for each estimate.

Burden tests often use various maximum minor allele frequency (MAF) filters. For instance,
[[Backman et al.](https://doi.org/10.1038/s41586-021-04103-z)] used MAF filters of 1%, 0.1%, 0.01%, 
and 0.001% [[2](https://doi.org/10.1038/s41586-021-04103-z)]. Presumably, such filters assume that 
increasingly stringent filters will reduce the number of false positive LoF variants that are 
aggregated into the burden genotype, as LoF variants that are at high frequency in the population 
might represent misannotated LoF variants. However, such filters result in asymmetric behavior 
across genes depending on their constraint. For example, the highly stringent 0.001% filter will 
reduce the false-positive rate in highly constrained genes, but will remove true LoF variants in 
unconstrained genes. Ideally, a gene under high constraint should use a stringent MAF filter, while 
a gene under low constraint should use a liberal MAF filter.

![](/static/posts/synonymous_variants/misannotation_probability_se.svg)

**Figure 4:** The mean standard error of burden estimates for genes in various buckets of 
$\log_{10} \left( s_{\mathrm{het}} \right)$ for various traits. Since larger-effect variants are 
under increased selective constraint, their frequencies are lower and the estimation noise is 
larger. All data here uses a MAF < 1% filter other than the MAF < 0.001% associations from
[[Backman et al.](https://doi.org/10.1038/s41586-021-04103-z)]. The use of the misannotation 
probability increases the noise of the estimates. The noise in the original analysis of the LoF data 
is also shown for reference. 95% confidence intervals are present on the plots but not visible due 
to their small length.

To account for this, [[Zeng et al.](https://doi.org/10.1038/s41588-024-01820-9)] have calculated 
misannotation probabilities for all potential LoF-introducing single nucleotide polymorphisms (SNPs) 
in genes for which they had estimated $s_{\mathrm{het}}$ values 
[[3](https://doi.org/10.1038/s41588-024-01820-9)]. We used various misannotation probability filters 
instead of MAF filters. We tested filters of 10%, 5%, and 1% misannotation probability for all 
variants with MAF < 1%. We estimated the total signal, as measured by the mean squared effect size, 
in various buckets of gene constraint (Figure 3). Increasingly stringent misannotation probability 
filters increased the signal across various selection buckets. In addition, the various filters were 
as or more effective than both the 1\% and 0.001% MAF filters from
[[2]((https://doi.org/10.1038/s41586-021-04103-z))].

Increasing stringency with the misannotation probability filters does increase the amount of 
estimation noise in $\hat{\gamma}_{\mathrm{LoF}}$ as more LoF variants are removed from the burden 
genotypes (Figure 4). We found that a misannotation probability filter of 10% provided signal 
comparable to or better than a MAF filter of 0.001% with a minimal increase in noise. This filter 
was used in all subsequent analyses.
