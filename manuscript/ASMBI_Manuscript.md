---
title: 'When More Trades Do Not Mean More Evidence: Selection Bias Induced by Minimum Trade Count Filters in Backtesting'
author: 'MIN HAN HSIEH · Independent researcher, Taiwan'
date: '26 November 2026'
---

# Abstract

Minimum completed-trade counts are frequently used to screen backtests for statistical reliability. When trade duration and reward are dependent, however, count eligibility within a fixed observation horizon changes the conditional distribution of completed trades. We present an exact marked-renewal identity for conditional empirical moments and a controlled synthetic experiment under a zero-mean complete-trade reward model. In a new 30,000-window simulation, the asymmetric exit rule with a minimum of 30 completed trades yielded 935 eligible windows out of 10,000. A one-sided Student t diagnostic rejected in 465 eligible windows (49.73%), compared with 147 (15.72%) for independent rewards matched to the same trade counts. The paired difference was 34.01 percentage points (95% percentile bootstrap interval: 29.95–38.07), while overall rejection frequencies were 4.65% and 1.47%. Other contrasts illustrate that the direction of the effect is not universal. We distinguish conditional rejection from overall rejection and discuss support loss and overlap limitations. Practical recommendations include retaining eligibility and validity denominators, separating event-time and calendar-time estimands, and calibrating the complete backtesting workflow under an explicit null. These findings are model-specific and do not establish real-market profitability or a universally valid count threshold.

**Keywords:** renewal-reward process; backtesting; informative selection; conditional inference.

# 1. Executive summary

A minimum number of completed trades is a reasonable operational eligibility rule for a backtest, but it is not a certificate of statistical reliability. When the duration and reward of trades are associated, a fixed observation window and a minimum count can select unusual outcomes. A larger count can therefore change the evidence rather than simply improve its precision.

In a historical synthetic Gaussian experiment with zero expected complete-cycle reward, 850 of 10,000 windows passed a 30-trade threshold. A conventional one-sided t diagnostic rejected in 401 of these windows (47.18%), versus 105 (12.35%) when the observed count was retained but rewards were drawn independently. Across all 10,000 windows the corresponding rejection frequencies were 4.01% and 1.05%. The distinction matters: a high rejection frequency among eligible results need not imply a high rejection frequency across the full procedure. The historical aggregate file is available, but its original simulator and path-level records have not been recovered. These historical figures are provisional because their original provenance is incomplete. A newly implemented simulation with fresh random draws yields 465 versus 147 rejections among 935 eligible windows (49.73% versus 15.72%), with overall frequencies of 4.65% and 1.47%. The new computation is the principal reproducible numerical illustration of this version.

We explain the mechanism using a marked renewal model, demonstrate why the direction of the effect need not be universal, and recommend practical reporting controls. Backtest systems should retain eligibility and validity denominators, distinguish event-time and calendar-time estimands, and calibrate the complete selection-and-testing pipeline under an explicit null. The contribution is an applied diagnostic framework, not a novel general renewal theorem or a universal correction.

# 2. Core section: problem, implementation and practical outcomes

## 2.1 Practical problem

Performance evaluation is often conducted over a fixed calendar window. In trading, a minimum number of transactions may be required before a strategy is considered for further analysis; for example, Sheppert ^1^ (Section 3.4) penalizes parameterizations generating fewer than 50 trades in a strategy-optimization study. Such requirements address a real concern: estimates calculated from a handful of outcomes are noisy, and ratios with estimated standard deviations can be especially unstable. The count requirement is nevertheless distinct from collecting a prespecified number of independent observations. A strategy may meet the requirement precisely because the events realized during the window have atypical durations and rewards.

Consider two exit rules with identical expected complete-trade rewards. One has a nearby profit barrier and a distant loss barrier. Short favorable trades can be numerous, whereas adverse trades may occupy more of the finite observation window. Conditional on observing many completed trades, the distribution of rewards may differ from the unconditional complete-trade distribution. It is not sufficient to reason that a larger observed count necessarily supplies more representative evidence. Moreover, even independent fixed-count rewards may have skewed distributions for which ordinary t approximations are inaccurate.

The renewal framework is closely related to budget-constrained simulation and cumulative-process limit theory ^2^. Glynn and Heidelberger ^3^ examine sample means and nonlinear estimators when the number of completed replications depends on a time budget. Window-censored observation of renewal processes, with its combination of censoring and length bias, is a classical estimation problem ^4^. Bao ^5^ studies the sampling bias and variance of the Sharpe ratio under general return distributions, and Kaminski and Lo ^6^ analyse the value of stop-loss rules under random-walk and other return processes. Backtest selection and multiplicity are also well-recognized concerns ^7^. Our purpose is to connect these issues at the level of an explicit operational eligibility rule, using count-matched controls to distinguish changes in count composition from the dependence between count and outcomes. We study the count threshold itself as a conditioning event, rather than proposing a new optimizer or a general renewal theorem. The cited work establishes the surrounding methodological context; we make no claim of priority for informative sampling, renewal truncation or inverse weighting.

The primary estimand is the population mean complete-event reward. For a sample of $n$ completed rewards with mean $\bar X$ and sample standard deviation $s$ (using denominator $n-1$), the one-sided diagnostic rejects when $\sqrt{n}\bar X/s$ exceeds the 95th percentile of the Student $t$ distribution with $n-1$ degrees of freedom. Let $V$ denote validity ($n\ge2$ and $s^2>10^{-10}$ in squared innovation-standard-deviation units), $A_m$ eligibility, and $R$ rejection, defined to be false outside eligibility or validity. Thus $\alpha_{\mathrm{all}}=P(R)=p_{\mathrm{valid}}\alpha_{\mathrm{cond}}$, where $p_{\mathrm{valid}}=P(A_m\cap V)$ and $\alpha_{\mathrm{cond}}=P(R\mid A_m\cap V)$ when $p_{\mathrm{valid}}>0$. Nominal 5% calibration is not assumed under skewness or selection. The sample Sharpe ratio is used as a familiar performance statistic, and the associated one-sided t statistic as a deliberately simple diagnostic test. Calendar-time performance is a distinct estimand. The study concerns statistical behavior under a specified synthetic null, not the profitability of a strategy, real-market forecasting or a universally safe threshold.

## 2.2 Experimental design and provenance

The historical experiment used a 250-day additive zero-drift Gaussian P&L model, profit/loss thresholds in units of daily innovation standard deviation, retained overshoots, a one-day flat interval after exit, and excluded terminal incomplete trades from event-time inference. Its primary historical contrast was TP=0.5, SL=10 and m=30, with 10,000 windows and reported root seed 20261125. The matched control retains the observed completed count while independently drawing complete-cycle rewards. Its aggregate results are available, but the historical code, path-level output and pre-run hash record are missing. We therefore distinguish this historical evidence from the newly implemented experiment.

The new script `experiments/replication_v2.py` uses independent standard Gaussian increments (equivalent in law to independently signed Gaussian innovations), the same additive barriers and flat interval, fresh root seed 20261126, 10,000 windows per cell by default, three cells (0.5/10, 3/3, 10/0.5), and thresholds 2, 20 and 30. It draws independent complete-cycle rewards directly, not from the historical estimated pool. It writes path-level outputs, summaries and a figure. It is a new implementation, not a recovery of the missing historical simulator. The initial execution on 26 November 2026 used NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and Matplotlib 3.10.8. A subsequent AI-assisted review reported reproduction of all summary values to six decimal places in a clean environment using the repository-pinned NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3 and Matplotlib 3.10.3. It completed 30,000 windows in total. Gaussian increments and rewards are expressed in units of the daily innovation standard deviation; the sample variance tolerance is 1e-10 in these units.

## 2.3 New simulation outcomes


Table 1 summarizes the nine new contrasts. Figure 1 visualizes the conditional and overall rejection frequencies for the asymmetric 0.5/10 cell. The primary illustrative contrast at m=30 in the 0.5/10 cell is 34.01 percentage points (paired percentile bootstrap interval 29.95–38.07; 20,000 bootstrap draws ^8^). The symmetric m=30 row has only one eligible window; its zero rejections and degenerate empirical bootstrap interval are not informative about population size. Other contrasts are descriptive; no multiplicity adjustment is claimed. The contrast was chosen based on earlier work, not preregistered. The new implementation supports the qualitative mechanism but is not an exact rerun of the historical implementation.

## 2.4 Historical aggregate outcomes


Table 2 reports the historical aggregate outcomes. For the first row the paired difference is 34.82 percentage points; conditional and overall frequencies have different denominators. The aggregate results also indicate that the direction is not uniform. The historical bootstrap interval is retained as reported, rather than claimed to have been recomputed from unavailable paths. Earlier exploratory results with unavailable source files, unverified tail moments, and secondary grid-search results are not used as evidence in this paper.

## 2.5 Practical workflow

An operational backtest should retain the records of candidates and windows before filtering, including completed counts, observation times, terminal position status and reasons for invalidity. The report should state whether its target is event reward, calendar P&L or a selected-strategy statistic. It should present eligibility and validity denominators, and conditional and unconditional rejection where a statistical test is used. For rare eligibility, uncertainty must be reported; a small number of accepted windows is not decisive evidence. Wilson score intervals ^9^ are one option for individual rejection proportions; paired comparisons require methods that retain the pairing.

Calibration should reproduce the complete evaluation procedure under a defensible null model, including terminal liquidation, selection rules and parameter searches. For a fixed candidate, a conditional null simulation may be appropriate when a sufficiently accurate model is available; for a searched strategy, calibration must reproduce the search and its dependence structure. Model-matched calibration cannot protect against arbitrary misspecification. Calendar-time P&L reporting complements trade-time metrics but is not a universal cure. Where inverse weighting is considered, report the eligibility probability, the support coverage and both the bulk (for example $v_{0.99}^Q$) and the tail ($K$) of the weight distribution before interpreting an adjusted estimate. Missing support requires assumptions or additional observations, not merely larger weights.

In the illustrative example, the count threshold can be useful for operational eligibility but should not be described as a statistical significance guarantee. The same logic may be relevant to fixed-window evaluation of sequential service, production or computational tasks if the iid marked renewal assumptions are appropriate. We have not empirically validated those other applications.

# 3. Statistical and stochastic methodology

## 3.1 Marked renewal representation

Let $(X_i,L_i)$, $i\ge 1$, be independent and identically distributed pairs, where $X_i$ is a complete-event reward and $L_i>0$ is a cycle duration. Dependence within a pair is allowed. Define $S_n=\sum_{i=1}^n L_i$, $S_0=0$, and $N_T=\max\{n:S_n\le T\}$. Write $F_j(t)=P(S_j\le t)$ with $F_0(t)=1\{t\ge0\}$. Let $A_m=\{N_T\ge m\}$, $m\ge1$, and assume $F_m(T)>0$. We use $T$ as a generic renewal horizon; the simulation uses a horizon of 251 to account for its post-exit flat day.

Since $A_m=\{S_m\le T\}$, independence across cycles gives the conditional first-event law

$$Q_m(dx,dl)=P((X_1,L_1)\in(dx,dl)\mid A_m)=\frac{w_m(l)}{F_m(T)}P(dx,dl),\qquad w_m(l)=F_{m-1}(T-l).$$

The weight is nonincreasing in duration and satisfies $E[w_m(L)]=F_m(T)$. It can be zero or very small for long cycles. This is a direct application of conditional probability. It is not a statement that all completed observations remain independent under the filter.

## 3.2 Exact expected empirical moments

**Proposition 1.** For any measurable function $g$ with $E|g(X,L)|<\infty$,

$$E\left[\frac{1}{N_T}\sum_{i=1}^{N_T}g(X_i,L_i)\mid A_m\right]=\frac{E[g(X,L)w_m(L)]}{F_m(T)}.$$

The proof uses exchangeability of the completed cycles at each realized count and is given in the Appendix. The case $m=1$ is related to the fixed-budget sample-mean identity in ^3^; the threshold extension is a direct conditioning argument.

For $g=X$, write $\mu_X=E[X]$, $h(l)=E[X\mid L=l]$, and define the conditional mean bias

$$B_m(T)=E[\bar X_{N_T}\mid A_m]-\mu_X=\frac{\operatorname{Cov}(X,w_m(L))}{F_m(T)},$$

using $E[w_m(L)]=F_m(T)$. If $h$ is nonincreasing, the covariance is nonnegative because $h$ and $w_m$ have the same monotonic direction. Specifically, for independent copies $L,L'$, twice their covariance equals the expectation of $(h(L)-h(L'))(w_m(L)-w_m(L'))$. Strict positivity requires this product to be positive with positive probability. A negative covariance between reward and duration alone is insufficient.

**Example 1.** Assign probability one third to each pair $(L,X)=(1,-1),(2,4),(3,-3)$. Then $E[X]=0$ and $\operatorname{Cov}(L,X)=-2/3$. For $T=2$, the conditional mean is 1.5 at $m=1$ but −1 at $m=2$. The latter threshold requires two unit-duration cycles. Thus negative duration–reward covariance guarantees neither a positive bias nor a bias that increases with the count threshold.

## 3.3 Comparing thresholds

On the positive-weight support of $Q_m$, define $r_m(l)=w_{m+1}(l)/w_m(l)$ and set it arbitrarily elsewhere. Provided $F_{m+1}(T)>0$, the next conditional law is $dQ_{m+1}=r_m\,dQ_m/E_{Q_m}[r_m]$, where $E_{Q_m}[r_m]=F_{m+1}(T)/F_m(T)$ is the conditional probability of passing the next threshold. It follows that

$$B_{m+1}(T)-B_m(T)=\frac{\operatorname{Cov}_{Q_m}(h(L),r_m(L))}{E_{Q_m}[r_m(L)]}.$$

The formula permits direct numerical evaluation of the incremental effect without assuming monotonicity. Monotonicity or log-concavity assumptions require separate justification for a given duration–reward law. The identities apply to empirical averages of individual-event functions, not to the expectation of a nonlinear ratio such as the sample Sharpe ratio. The dependence induced by the filter cannot be removed by substituting the realized count into an iid sampling formula.

## 3.4 Support loss and overlap

The duration weight identifies the region of the event law with positive probability of inclusion. If $L\ge\ell_*>0$ almost surely, every completed event in an eligible window satisfies $L_i\le T-(m-1)\ell_*$, because $L_i+(N_T-1)\ell_*\le S_{N_T}\le T$; for $m>1$ an event of exactly that upper-bound duration can have positive conditional mass only if the other $m-1$ cycles can all have length $\ell_*$. More precisely the positive-weight region is $H=\{l:w_m(l)>0\}$ up to null sets.

Two models with identical duration distributions and identical reward laws conditional on duration in $H$ generate identical retained completed-path distributions even if their reward laws outside $H$ differ. For example, let $T=3$, $m=2$ and let $L=1$ or 3 with probability one half. Short rewards equal 1. If long rewards equal −1 in one model and 3 in another, the population means are 0 and 2, but the retained data are observationally equivalent. No test based only on these data can have power greater than its size against this particular indistinguishable alternative. The result is a support-based limitation, not a universal claim about parametric exit models or datasets containing additional information. Whether the missing mass itself can be recovered from retained paths depends on the design; for instance, with $T=2$, $m=2$ and $L\in\{1,3\}$, every eligible window contains exactly two unit cycles regardless of $P(L=1)$, so the missing mass is not identified from retained paths alone.

When the duration law is known, the missing mass $q=P(L\notin H)$ and the observable contribution $\mu_H=E[X1\{L\in H\}]$ yield the sharp interval $[\mu_H+qA,\mu_H+qB]$ if reward is known to lie in $[A,B]$ and the unseen conditional reward law is unrestricted. This is the standard worst-case bound for selectively observed outcomes ^10^; confidence intervals for such interval-identified parameters are available ^11^. For unbounded rewards with known $E[X^2]\le M_2$, Cauchy–Schwarz instead bounds the absolute missing contribution by $\sqrt{qM_2}$. Discrete Gaussian exits have unbounded overshoots, so empirical minima and maxima must not be treated as known population bounds.

Even small $q$ does not assure a stable weighting correction. On $H$ let $v=F_m(T)/w_m(L)$. The conditional marginal distribution $Q$ satisfies

$$E_Q[v]=1-q,\qquad E_Q[v^2]=F_m(T)E_P\left[\frac{1\{L\in H\}}{w_m(L)}\right].$$

A convenient diagnostic is the Kish-style marginal weight ratio $K=(1-q)^2/E_Q[v^2]$ ^12^, which lies between zero and one. It is a marginal weight-concentration measure, not an estimate of the number of independent paths. Large inverse weights can amplify tail uncertainty, as related support and inverse-weight research demonstrates ^13^. In the present fixed discrete horizon, positive weights lie on a finite support and have a positive minimum for a fixed law. Numerical instability therefore does not by itself establish irregular asymptotic identification or a slower-than-root-n rate.

The historical overlap summary is available in `data/ENSI_round4_corrected_summary.csv`. We do not report unavailable quantiles, sub-pool ranges or reward-weighted tail moments as verified evidence.

# 4. Summary and discussion

Minimum-count filtering is a selection operation, not simply a sample-size requirement. The renewal identity shows how the event law is reweighted. The matched-count comparison is designed to separate the coupling of duration and reward from fixed-count sampling effects. Conditional rejection among eligible windows and rejection across the entire pipeline must be reported separately.

The synthetic model has no transaction costs, serial dependence, real-market validation or genuine positive-reward alternatives. It does not establish profitability, universal bias direction, power or a universal threshold. Historical results are limited by the missing original code and paths; the new experiment is a separate computational check. The method may be relevant to other fixed-window sequential tasks, but those applications have not been empirically validated. The new simulation code and numerical summary are available in the public research repository ^14^.

# Declarations

**Data and code availability.** Historical aggregate results and a newly specified replication implementation are publicly available at https://github.com/salamon-aska/trade-count-selection-bias. The historical path-level data and original simulator have not been recovered; the new implementation must not be represented as reproducing the historical numerical values. The new summary (`summary.csv`), 30,000 per-window records (`paths.csv.gz`), and original rejection-rate figure (`rejection_rates.png`) are publicly archived in the repository under `data/recomputed/`. To regenerate them, install the pinned requirements and run `python experiments/replication_v2.py --out data/recomputed --n 10000 --seed 20261126` from the repository root. Figure 1 is generated directly from the archived summary using `figures/make_submission_figure.py`. The per-window records and figure are also included in the accompanying replication archive. The historical path-level records remain unavailable; historical aggregate CSV files are separately identified.

**Funding.** This independent research received no external research funding.

**Conflicts of interest.** The author declares no known conflicts of interest. The work was conducted as independent personal research.

**Author contributions.** MIN HAN HSIEH directed the research, formulated the research questions and made the research decisions, with substantial AI assistance as described below. The author independently verified the mathematical derivations, source code, numerical results, references and manuscript content, and approved the manuscript for submission.

**Generative AI disclosure.** OpenAI ChatGPT and Anthropic Claude provided substantial assistance with research design, literature exploration, mathematical derivations and counterexamples, code development and review, interpretation of numerical outputs, and manuscript drafting and revision. The tools also assisted with academic English editing and preparation of submission materials. The tools were used in conversational research, coding and review sessions. Exact historical model identifiers have not been fully documented. Numerical results were obtained by executing explicitly specified stochastic simulation and statistical analysis code developed with substantial generative AI assistance; numerical observations were not directly sampled from language-model responses. Neither AI system is an author. The author is responsible for the accuracy, originality and integrity of the work. On 26 November 2026, the author confirmed completion of independent human verification of the mathematical derivations, code, numerical results, references and manuscript content, accepted full authorship responsibility, and confirmed compliance with Wiley's publication ethics and AI disclosure requirements. AI-assisted re-execution and review do not substitute for human verification.

**Ethics and consent.** This study uses synthetic simulated data and does not involve human participants, animals, personal data or clinical trials; participant consent and research ethics approval are not applicable.

**Supporting information.** Additional information for this article is available online: a replication archive containing the new simulation and validation code, dependency specifications, aggregate results, per-window records, figure-generation code, and separately labelled historical aggregate summaries.

# References

1. Sheppert, A. P. 2026. The GT-Score: A Robust Objective Function for Reducing Overfitting in Data-Driven Trading Strategies. *Journal of Risk and Financial Management* 19(1): 60. https://doi.org/10.3390/jrfm19010060

2. Glynn, P. W., and W. Whitt. 1993. Limit Theorems for Cumulative Processes. *Stochastic Processes and their Applications* 47(2): 299–314. https://doi.org/10.1016/0304-4149(93)90019-Z

3. Glynn, P. W., and P. Heidelberger. 1990. Bias Properties of Budget Constrained Simulations. *Operations Research* 38(5): 801–814. https://doi.org/10.1287/opre.38.5.801

4. Vardi, Y. 1982. Nonparametric Estimation in Renewal Processes. *Annals of Statistics* 10(3): 772–785. https://doi.org/10.1214/aos/1176345870

5. Bao, Y. 2009. Estimation Risk-Adjusted Sharpe Ratio and Fund Performance Ranking under a General Return Distribution. *Journal of Financial Econometrics* 7(2): 152–173. https://doi.org/10.1093/jjfinec/nbn022

6. Kaminski, K. M., and A. W. Lo. 2014. When Do Stop-Loss Rules Stop Losses? *Journal of Financial Markets* 18: 234–254. https://doi.org/10.1016/j.finmar.2013.07.001

7. Bailey, D. H., and M. López de Prado. 2014. The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management* 40(5): 94–107. https://doi.org/10.3905/jpm.2014.40.5.094

8. Efron, B., and R. J. Tibshirani. 1993. *An Introduction to the Bootstrap*. Chapman & Hall. https://doi.org/10.1201/9780429246593

9. Wilson, E. B. 1927. Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association* 22(158): 209–212. https://doi.org/10.1080/01621459.1927.10502953

10. Manski, C. F. 1989. Anatomy of the Selection Problem. *Journal of Human Resources* 24(3): 343–360. https://doi.org/10.2307/145818

11. Imbens, G. W., and C. F. Manski. 2004. Confidence Intervals for Partially Identified Parameters. *Econometrica* 72(6): 1845–1857. https://doi.org/10.1111/j.1468-0262.2004.00555.x

12. Kish, L. 1965. *Survey Sampling*. New York: John Wiley & Sons.

13. Khan, S., and E. Tamer. 2010. Irregular Identification, Support Conditions, and Inverse Weight Estimation. *Econometrica* 78(6): 2021–2042. https://doi.org/10.3982/ECTA7372

14. Hsieh, M. H. 2026. When More Trades Do Not Mean More Evidence: Selection Bias Induced by Minimum Trade Count Filters in Backtesting. Data and code repository. GitHub. https://github.com/salamon-aska/trade-count-selection-bias/tree/05d3bdd3303af493ccdb364c7cc9becd6f81f8e0

# Appendix. Proofs and integrability

**Proposition 1.** Proof. For every $n\ge m$, the event $\{N_T=n\}=\{S_n\le T<S_n+L_{n+1}\}$ is invariant under permutations of the first $n$ pairs. Thus the expectation of their average multiplied by the event indicator equals the expectation of the first term multiplied by that indicator. Summing over $n\ge m$ yields $E[g(X_1,L_1);A_m]$. Applying the conditional law above and dividing by $P(A_m)$ proves the identity. For absolute integrability, first apply the argument to $|g|$ using Tonelli's theorem. The case $m=1$ is the fixed-budget sample-mean identity discussed in ^3^; the extension to $m\ge2$ is immediate.

For the increment identity, $Q_{m+1}$ is absolutely continuous with respect to $Q_m$ because $w_{m+1}\le w_m$. Its Radon–Nikodym derivative is $r_m/E_{Q_m}[r_m]$. Subtracting the two expectations of $h(L)$ gives the stated covariance identity. The conditional empirical moments need not imply moments of a ratio estimator.

Under the Gaussian null, the complete-trade exit time has a geometric tail bound: from any position inside the finite interval, an increment larger than the full barrier width in absolute value in either specified direction has a fixed positive probability of causing exit. Thus the exit time has finite expectation. For the zero-drift standard Gaussian random walk $M_j$, the stopped-martingale second-moment identity gives $E[M_{\tau\wedge n}^2]=E[\tau\wedge n]\le E[\tau]<\infty$. This uniform $L^2$ bound implies uniform integrability of the stopped sums. Since $M_{\tau\wedge n}\to M_\tau$ almost surely, convergence of expectations gives $E[M_\tau]=0$. This justifies the null estimand used here.

# Tables

**Table 1.** Newly executed simulation, 10,000 windows per cell, seed 20261126. A: completed-trade diagnostic; E: independent count-matched diagnostic. Joint valid counts equal eligibility counts in these rows. Source: `data/recomputed/summary.csv`.

| TP/SL | m | Eligible | A rejects | E rejects | A conditional (%) | E conditional (%) | A overall (%) | E overall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0.5/10|2|9998|1690|1884|16.90|18.84|16.90|18.84|
|0.5/10|20|4985|1049|609|21.04|12.22|10.49|6.09|
|0.5/10|30|935|465|147|49.73|15.72|4.65|1.47|
|3/3|2|10000|516|471|5.16|4.71|5.16|4.71|
|3/3|20|2759|148|125|5.36|4.53|1.48|1.25|
|3/3|30|1|0|0|0.00|0.00|0.00|0.00|
|10/0.5|2|9998|43|135|0.43|1.35|0.43|1.35|
|10/0.5|20|5009|5|88|0.10|1.76|0.05|0.88|
|10/0.5|30|920|0|21|0.00|2.28|0.00|0.21|

**Table 2.** Historical independent confirmation (10,000 windows per cell). A and E denote completed trades and count-matched independent rewards. Joint valid counts equal eligible counts in all displayed rows; denominators of individual rates are shown. Paired difference and 95% percentile bootstrap interval are in percentage points; overall rates have denominator 10,000. The first row is primary; remaining rows are descriptive. Source: `data/ENSI_confirmation_summary.csv` (historical aggregate output; path-level records not recovered).

| TP/SL | m | Eligible | A rejects / valid (%) | E rejects / valid (%) | Paired difference [95% CI] | Overall A/E (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5/10 | 30 | 850 | 401/850 (47.18%) | 105/850 (12.35%) | 34.82 [30.82, 38.82] | 4.01 / 1.05 |
| 0.5/10 | 2 | 9,998 | 1699/9998 (16.99%) | 1883/9998 (18.83%) | -1.84 [-2.91, -0.77] | 16.99 / 18.83 |
| 0.5/10 | 20 | 5,081 | 1050/5081 (20.67%) | 592/5081 (11.65%) | 9.01 [7.60, 10.45] | 10.50 / 5.92 |
| 3/3 | 20 | 2,743 | 132/2743 (4.81%) | 147/2743 (5.36%) | -0.55 [-1.71, 0.62] | 1.32 / 1.47 |
| 10/0.5 | 30 | 965 | 0/965 (0.00%) | 23/965 (2.38%) | -2.38 [-3.42, -1.45] | 0.00 / 0.23 |

# Figure legend

**Figure 1.** Rejection frequencies from the new simulation for TP/SL = 0.5/10. Panel (a) uses the eligible and numerically valid windows; panel (b) uses all 10,000 windows. The completed-trade diagnostic (A) and independent count-matched control (E) are shown at minimum counts 2, 20 and 30. The horizontal dotted line is the nominal 5% reference, not a guarantee of calibration. The figure is a visualization of Table 1 and does not constitute additional experimental evidence.
