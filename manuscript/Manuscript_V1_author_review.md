<!-- Public pre-submission author-review draft; not accepted or independently verified. -->

---
title: 'When More Trades Do Not Mean More Evidence: Selection Bias Induced by Minimum Trade Count Filters in Backtesting'
subtitle: 'Practitioner Paper | Applied Stochastic Models in Business and Industry'
author: '[Confirm full publishing name] · Independent researcher, Taiwan'
date: '26 November 2026'
---

# Executive summary

A minimum number of completed trades is an intuitive eligibility criterion for a backtest. For example, Sheppert [6] uses a 50-trade minimum in a published strategy-optimization study. A strategy with only a few observed trades may indeed produce an unstable performance estimate. Increasing the minimum count, however, does not necessarily make the retained results more representative. When trade duration and profit are associated, a fixed backtesting horizon and a minimum completed-trade count jointly select particular realizations of the underlying trading process.

We illustrate this in a controlled, zero-drift Gaussian experiment. All candidate strategies have zero expected complete-cycle reward under the model, but asymmetric exit barriers produce rewards of unequal magnitude and duration. Among 10,000 simulated windows for one asymmetric candidate, 904 meet a minimum of 30 completed trades. A conventional one-sided t test rejects in 427 of these windows (47.2%; 95% Wilson interval 44.0–50.5%), compared with 152 of 904 (16.8%; 14.5–19.4%) in a control with the same counts but independent complete-cycle rewards. The unconditional rejection frequency of the filter-and-test procedure is 4.27%. An independent 10,000-window confirmation retains 850 windows and yields 401 versus 105 rejections (47.18% versus 12.35%). The paired difference is 34.82 percentage points (95% bootstrap interval 30.82–38.82); the overall rates are 4.01% and 1.05%. These two error measures answer different questions: the first concerns inference among eligible results; the second concerns the entire procedure.

A marked renewal model explains why the filter changes the distribution of completed trades. It gives exact expected empirical moments, a counterexample to a universal directional claim, and diagnostics for missing support and unstable inverse weights. These calculations use established probability and statistical principles rather than a new general theory. The practical recommendation is to treat count eligibility as a selection operation: preserve its denominator, distinguish trade and calendar clocks, disclose conditional and unconditional rejection, and calibrate the complete evaluation workflow under an explicitly stated null. Minimum counts alone do not certify statistical reliability.

# Abstract

Minimum trade counts are commonly motivated by concerns about unstable small-sample performance estimates. We examine a less apparent consequence of count filtering when event durations and rewards are dependent and observation time is fixed. A marked renewal representation yields exact conditional empirical-moment identities and shows that count eligibility reweights the event law. Gaussian first-exit experiments compare completed-trade inference with fixed-count, count-matched independent and calendar-time controls. In an independent confirmation of an asymmetric exit rule with a 30-trade threshold, conditional rejection under a zero-reward null is 47.18% versus 12.35% in a matched independent-reward control (850 eligible windows; paired difference 34.82 percentage points, 95% bootstrap interval 30.82–38.82). Overall rejection is 4.01% versus 1.05%. Inverse-weight diagnostics show that a very small missing-support mass can coexist with an extreme second moment of the weights, driven by rare long durations rather than by the bulk of retained events. We discuss the limitations of normal approximations and propose a reporting and calibration workflow for event-based performance evaluation. The contribution is an applied illustration and diagnostic framework, not a claim of a novel renewal theorem or a universally valid correction.

**Keywords:** renewal-reward process; backtesting; informative selection; Sharpe ratio; limited overlap; conditional inference.

# 1. Introduction

Performance evaluation is often conducted over a fixed calendar window. In trading, a minimum number of transactions may be required before a strategy is considered for further analysis; for example, Sheppert [6, Section 3.4] penalizes parameterizations generating fewer than 50 trades in a strategy-optimization study. Such requirements address a real concern: estimates calculated from a handful of outcomes are noisy, and ratios with estimated standard deviations can be especially unstable. The count requirement is nevertheless distinct from collecting a prespecified number of independent observations. A strategy may meet the requirement precisely because the events realized during the window have atypical durations and rewards.

Consider two exit rules with identical expected complete-trade rewards. One has a nearby profit barrier and a distant loss barrier. Short favorable trades can be numerous, whereas adverse trades may occupy more of the finite observation window. Conditional on observing many completed trades, the distribution of rewards may differ from the unconditional complete-trade distribution. It is not sufficient to reason that a larger observed count necessarily supplies more representative evidence. Moreover, even independent fixed-count rewards may have skewed distributions for which ordinary t approximations are inaccurate.

The renewal framework is closely related to budget-constrained simulation. Glynn and Heidelberger [1] examine sample means and nonlinear estimators when the number of completed replications depends on a time budget. Window-censored observation of renewal processes, with its combination of censoring and length bias, is a classical estimation problem [7]. Bao [2] studies the sampling bias and variance of the Sharpe ratio under general return distributions, and Kaminski and Lo [8] analyse the value of stop-loss rules under random-walk and other return processes. Backtest selection and multiplicity are also well-recognized concerns [3]. Our purpose is to connect these issues at the level of an explicit operational eligibility rule, using count-matched controls to distinguish changes in count composition from the dependence between count and outcomes. We study the count threshold itself as a conditioning event, rather than proposing a new optimizer or a general renewal theorem. The cited work establishes the surrounding methodological context; we make no claim of priority for informative sampling, renewal truncation or inverse weighting.

The primary estimand is the population mean complete-event reward. The sample Sharpe ratio is used as a familiar performance statistic, and the associated one-sided t statistic as a deliberately simple diagnostic test. Calendar-time performance is a distinct estimand. The study concerns statistical behavior under a specified synthetic null, not the profitability of a strategy, real-market forecasting or a universally safe threshold.

# 2. Model and mechanism

## 2.1 Marked renewal representation

Let $(X_i,L_i)$, $i\ge 1$, be independent and identically distributed pairs, where $X_i$ is a complete-event reward and $L_i>0$ is a cycle duration. Dependence within a pair is allowed. Define $S_n=\sum_{i=1}^n L_i$, $S_0=0$, and $N_T=\max\{n:S_n\le T\}$. Write $F_j(t)=P(S_j\le t)$ with $F_0(t)=1\{t\ge0\}$. Let $A_m=\{N_T\ge m\}$, $m\ge1$, and assume $F_m(T)>0$. We use $T$ as a generic renewal horizon; the simulation uses a horizon of 251 to account for its post-exit flat day.

Since $A_m=\{S_m\le T\}$, independence across cycles gives the conditional first-event law

$$Q_m(dx,dl)=P((X_1,L_1)\in(dx,dl)\mid A_m)=\frac{w_m(l)}{F_m(T)}P(dx,dl),\qquad w_m(l)=F_{m-1}(T-l).$$

The weight is nonincreasing in duration and satisfies $E[w_m(L)]=F_m(T)$. It can be zero or very small for long cycles. This is a direct application of conditional probability. It is not a statement that all completed observations remain independent under the filter.

## 2.2 Exact expected empirical moments

**Proposition 1.** For any measurable function $g$ with $E|g(X,L)|<\infty$,

$$E\left[\frac{1}{N_T}\sum_{i=1}^{N_T}g(X_i,L_i)\mid A_m\right]=\frac{E[g(X,L)w_m(L)]}{F_m(T)}.$$

The proof uses exchangeability of the completed cycles at each realized count and is given in Appendix B. The case $m=1$ is related to the fixed-budget sample-mean identity in [1]; the threshold extension is a direct conditioning argument.

For $g=X$, write $\mu_X=E[X]$, $h(l)=E[X\mid L=l]$, and define the conditional mean bias

$$B_m(T)=E[\bar X_{N_T}\mid A_m]-\mu_X=\frac{\operatorname{Cov}(X,w_m(L))}{F_m(T)},$$

using $E[w_m(L)]=F_m(T)$. If $h$ is nonincreasing, the covariance is nonnegative because $h$ and $w_m$ have the same monotonic direction. Specifically, for independent copies $L,L'$, twice their covariance equals the expectation of $(h(L)-h(L'))(w_m(L)-w_m(L'))$. Strict positivity requires this product to be positive with positive probability. A negative covariance between reward and duration alone is insufficient.

**Example 1.** Assign probability one third to each pair $(L,X)=(1,-1),(2,4),(3,-3)$. Then $E[X]=0$ and $\operatorname{Cov}(L,X)=-2/3$. For $T=2$, the conditional mean is 1.5 at $m=1$ but −1 at $m=2$. The latter threshold requires two unit-duration cycles. Thus negative duration–reward covariance guarantees neither a positive bias nor a bias that increases with the count threshold.

## 2.3 Comparing thresholds

On the positive-weight support of $Q_m$, define $r_m(l)=w_{m+1}(l)/w_m(l)$ and set it arbitrarily elsewhere. Provided $F_{m+1}(T)>0$, the next conditional law is $dQ_{m+1}=r_m\,dQ_m/E_{Q_m}[r_m]$, where $E_{Q_m}[r_m]=F_{m+1}(T)/F_m(T)$ is the conditional probability of passing the next threshold. It follows that

$$B_{m+1}(T)-B_m(T)=\frac{\operatorname{Cov}_{Q_m}(h(L),r_m(L))}{E_{Q_m}[r_m(L)]}.$$

The formula permits direct numerical evaluation of the incremental effect without assuming monotonicity. In particular, the discrete Gaussian exit model can violate simple monotonicity or log-concavity assumptions. The identities apply to empirical averages of individual-event functions, not to the expectation of a nonlinear ratio such as the sample Sharpe ratio. The dependence induced by the filter cannot be removed by substituting the realized count into an iid sampling formula.

# 3. Experimental design

## 3.1 Synthetic trading environment

The exploratory experiment uses 250 independent Gaussian daily increments with mean zero and standard deviation 0.01 in units of fixed reference notional. This is an additive P&L model, not a positive-price market simulator. At entry a direction is drawn from a pre-generated daily random-sign sequence shared by all candidates, independent of later increments. A trade exits at the close at which cumulative signed P&L reaches or crosses a profit or loss barrier; overshoots are retained. A one-day flat interval follows each exit before re-entry. Transaction costs are zero. A position still open at the end of the horizon may be liquidated for terminal-inclusive reporting; it is excluded from the completed-cycle analysis. For completed cycles, if $\tau$ is the number of held increments, set $L=\tau+1$ and use renewal horizon 251. Under the specified zero-drift model, complete-cycle reward has mean zero, subject to the usual integrability conditions for optional stopping; the daily signed exposure is predictable.

Both profit and loss thresholds range over 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8 and 10 daily standard deviations, giving 100 candidate cells. The original study generates 10,000 calibration windows and 10,000 evaluation windows using separate child streams of seed 20261124. The V0.3 study deliberately reuses the V0.2 windows, so evidence from those windows is exploratory. A separate confirmation, described in Section 3.3, uses a new root seed and does not reuse those windows.

## 3.2 Controls and estimands

Four comparisons are particularly useful. A fixed candidate is evaluated from its completed-trade rewards (A) and from all 250 calendar-time P&L increments, including flat zeros (B). A fixed-count experiment (E) generates independent complete trades with sample sizes 10, 20, 50 and 100; the smaller samples are the first 10, 20 or 50 trades of each 100-trade sample, so the four sizes are nested rather than independent. A count-matched control (E-matched) preserves each A window's actual completed count, but replaces its rewards with independent draws from the complete-cycle law, independent of that count. The matched control covers seven representative cells. This design preserves the marginal count distribution and reward law while removing their within-window coupling. Because A and E-matched share the same windows and counts, their comparison is paired. The comparison is specific to the null model, not a causal claim about arbitrary trading data.

For valid samples with sample standard deviation $s>0$, the unannualized sample Sharpe ratio is $\bar X/s$. The primary diagnostic rejects when $\sqrt n\bar X/s>t_{n-1,0.95}$. This is not asserted to be an exact test under skewness or selection. The simulation uses a sample-variance tolerance of $10^{-14}$ for A and a sample-standard-deviation threshold of $10^{-7}$ for the matched control (the same variance scale), and requires at least two observations. For fixed-candidate analyses the minimum completed count is varied over 2, 5, 10, 20 and 30. The calendar and event clocks measure different quantities; their unannualized Sharpe values should not be compared as if they had identical units.

Let $V$ indicate a numerically valid statistic and $R$ indicate rejection, defined to be false outside eligibility and validity. We separately report

$$p_{\mathrm{pass}}=P(A_m),\quad p_{\mathrm{valid}}=P(A_m\cap V),\quad \alpha_{\mathrm{cond}}=P(R\mid A_m\cap V),\quad \alpha_{\mathrm{all}}=P(R).$$

Then $\alpha_{\mathrm{all}}=p_{\mathrm{valid}}\alpha_{\mathrm{cond}}$ whenever $p_{\mathrm{valid}}>0$. A conditional rate is undefined if there are no valid observations. In every displayed row of Tables 1 and 4 the eligible and valid counts coincide (no eligible samples were excluded for numerical invalidity). Conditional size under an all-null experiment is not the false discovery rate in a mixture of null and non-null candidates. The secondary grid-search experiment in Appendix A separately studies selection of the largest Sharpe statistic.

## 3.3 Independent confirmation

After the exploratory comparisons, a separate protocol fixed the primary contrast at TP=0.5, SL=10 (units of daily innovation SD), 250 daily increments, renewal horizon 251 and $m=30$. Secondary contrasts were the same cell at $m=2,20$, the symmetric 3/3 cell at $m=20$, and the reverse 10/0.5 cell at $m=30$. The protocol fixes 10,000 new windows, root seed 20261125 (distinct from the exploratory root seed), and separate child streams for windows, each independent complete-reward pool, and interval calculation. The implementation reuses the V0.3 timing and exit functions with only the three prespecified candidate cells. Code hashes were recorded before the new paths were generated. This is a within-model confirmation using fresh random draws, not a preregistered study or external-market validation.

For each eligible window retain the paired rejection indicators $R_A,R_E$ and define the primary contrast $\Delta=E[R_A-R_E\mid A_m\cap V_A\cap V_E]$. For each method also report its own eligibility/validity and rejection counts; failures outside eligibility or validity count as non-rejections for overall rates. The primary 95% interval is a percentile bootstrap of the mean paired difference, with 20,000 resamples of eligible jointly valid paths [11]. Resampling the empirical three-category difference distribution (−1, 0, +1) is equivalent to resampling paired paths for this statistic. Individual rates use Wilson intervals [10]. Secondary comparisons are descriptive and not multiplicity-adjusted. All prespecified outcomes are reported.

# 4. Results

## 4.1 Fixed-count instability and matched-count comparison

The fixed-count control already shows substantial departures from nominal size. Across the 100 cells, the maximum observed rejection frequencies are approximately 36.2%, 15.1%, 12.3% and 8.4% at counts 10, 20, 50 and 100, respectively. These maxima are descriptive, not simultaneous confidence bounds. The fixed-candidate calendar-time rates range from approximately 4.47% to 5.51% in this particular model. At zero mean, the known leading fixed-count Sharpe bias is $-\gamma_1/(2n)$ under suitable moment and denominator conditions [2]. Neither that approximation nor the calendar result guarantees valid inference for arbitrary market processes.

Table 1 and Figure 1 show the exploratory matched-count comparison. The asymmetric 0.5/10 cell shows similar rates with minimal filtering, but a pronounced separation at a threshold of 30. The 1/10 result at 30 has only 43 valid windows; its interval is wide and it should be interpreted cautiously. The symmetric and reverse-asymmetric cells illustrate that the effect is not uniformly positive.


The exploratory contrast at m=30 is 30.4 percentage points. Individual Wilson intervals do not account for the pairing; the separate confirmation below supplies a paired interval. The 427 rejections among 904 eligible and valid windows imply an overall rejection frequency of 4.27%, not 47.23%. By contrast, the same candidate at m=20 has an overall rate of 10.18% (95% Wilson interval 9.60–10.79%). Consequently, filtering can increase conditional rejection while decreasing overall rejection; it is incorrect either to equate the two or to claim that filtering never inflates overall error. The matched comparison also shows that count conditioning is not the sole source of inaccurate inference: skewed fixed-count samples can themselves produce elevated rejection.



## 4.2 Checking the conditional mean identity

An independent pool of two million complete-cycle draws per selected cell estimates the joint duration–reward law. Discrete convolutions then evaluate the conditional mean formula. This is a plug-in computation, not exact integration of the Gaussian first-exit law. The numerical comparison for the 0.5/10 cell is shown in Table 2. Agreement across 30 cell-threshold combinations supports the implementation; the reported window standard errors omit uncertainty in the estimated cycle law. They therefore do not constitute a formal simultaneous validation test.


At higher thresholds the conditional mean is positive despite a zero-mean population model. The pool has a small nonzero empirical mean; the table reports the conditional mean rather than subtracting that empirical baseline. The results illustrate the reweighting mechanism in Section 2. The experiment does not prove monotonicity across all possible exit rules.

## 4.3 Independent confirmation of the prespecified contrast

The fresh windows reproduce the key qualitative pattern (Table 4). In the primary cell, 850 of 10,000 windows are eligible and all 850 are numerically valid in both designs. Completed-trade inference rejects in 401 windows (47.18%; 95% Wilson interval 43.84–50.54%), whereas independent count-matched rewards reject in 105 (12.35%; 10.31–14.74%). Among paired windows, 356 reject only under A and 60 only under E; the paired difference is 34.82 percentage points (95% bootstrap interval 30.82–38.82). The overall frequencies are 4.01% and 1.05%. This supports the specific count-conditioning mechanism without implying an overall type-I error above 5% for this threshold. At $m=20$ the overall rate of A is 10.50%; at $m=2$ the independent control rejects more frequently. The symmetric and reversed controls do not support a universal increase from filtering. No alternative seeds or primary contrasts were selected based on outcomes.


# 5. Reliability diagnostics and practical implementation

## 5.1 Support loss and limited overlap

The duration weight identifies the region of the event law with positive probability of inclusion. If $L\ge\ell_*>0$ almost surely, every completed event in an eligible window satisfies $L_i\le T-(m-1)\ell_*$, because $L_i+(N_T-1)\ell_*\le S_{N_T}\le T$; for $m>1$ an event of exactly that upper-bound duration can have positive conditional mass only if the other $m-1$ cycles can all have length $\ell_*$. More precisely the positive-weight region is $H=\{l:w_m(l)>0\}$ up to null sets.

Two models with identical duration distributions and identical reward laws conditional on duration in $H$ generate identical retained completed-path distributions even if their reward laws outside $H$ differ. For example, let $T=3$, $m=2$ and let $L=1$ or 3 with probability one half. Short rewards equal 1. If long rewards equal −1 in one model and 3 in another, the population means are 0 and 2, but the retained data are observationally equivalent. No test based only on these data can have power greater than its size against this particular indistinguishable alternative. The result is a support-based limitation, not a universal claim about parametric exit models or datasets containing additional information. Whether the missing mass itself can be recovered from retained paths depends on the design; for instance, with $T=2$, $m=2$ and $L\in\{1,3\}$, every eligible window contains exactly two unit cycles regardless of $P(L=1)$, so the missing mass is not identified from retained paths alone.

When the duration law is known, the missing mass $q=P(L\notin H)$ and the observable contribution $\mu_H=E[X1\{L\in H\}]$ yield the sharp interval $[\mu_H+qA,\mu_H+qB]$ if reward is known to lie in $[A,B]$ and the unseen conditional reward law is unrestricted. This is the standard worst-case bound for selectively observed outcomes [5]; confidence intervals for such interval-identified parameters are available [9]. For unbounded rewards with known $E[X^2]\le M_2$, Cauchy–Schwarz instead bounds the absolute missing contribution by $\sqrt{qM_2}$. Discrete Gaussian exits have unbounded overshoots, so empirical minima and maxima must not be treated as known population bounds.

Even small $q$ does not assure a stable weighting correction. On $H$ let $v=F_m(T)/w_m(L)$. The conditional marginal distribution $Q$ satisfies

$$E_Q[v]=1-q,\qquad E_Q[v^2]=F_m(T)E_P\left[\frac{1\{L\in H\}}{w_m(L)}\right].$$

A convenient diagnostic is the Kish-style marginal weight ratio $K=(1-q)^2/E_Q[v^2]$ [12], which lies between zero and one. It is a marginal weight-concentration measure, not an estimate of the number of independent paths. Large inverse weights can amplify tail uncertainty, as related support and inverse-weight research demonstrates [4]. In the present fixed discrete horizon, positive weights lie on a finite support and have a positive minimum for a fixed law. Numerical instability therefore does not by itself establish irregular asymptotic identification or a slower-than-root-n rate.




Table 3 and Figure 2 summarize the marginal overlap diagnostics. The tiny estimated $q$ in the 251-horizon example coexists with an extreme second moment of the weights. That second moment is driven by rare long durations. A separate recomputation from the same two-million-cycle empirical law evaluates the share of the second moment above the $Q$-weighted 99th percentile, $E_Q[v^2 1\{v>v_{0.99}^Q\}]/E_Q[v^2]$. At horizon 251 and $m=30$, this tail contains 0.9995% of $Q$ mass and contributes more than 99.99% of $E_Q[v^2]$. At $m=20$ the corresponding share is 99.9957%. The analogous reward-weighted second moments $E_Q[v^2(X/\sigma)^2]$ are approximately $2.77\times10^{10}$ and $1.62\times10^6$, respectively, with more than 99.99% contributed by the same tails (tail_diagnostics.csv). These are plug-in marginal moments, not the variance of a path-level estimator; estimation of the weights, within-path dependence, and missing-support bias are additional issues. The order of magnitude of $K$ is stable across sub-pools, but no accuracy claim is made beyond that. Trimming such weights would stabilize an adjusted estimate at the price of a bias of unknown size. Extending the horizon improves the diagnostics in this cell; an observed zero tail count at horizon 501 does not establish a population zero. Other cells may behave differently. We do not propose a universal threshold based on expected event count or eligibility probability alone.

## 5.2 A reporting and calibration workflow

An operational backtest should retain the records of candidates and windows before filtering, including completed counts, observation times, terminal position status and reasons for invalidity. The report should state whether its target is event reward, calendar P&L or a selected-strategy statistic. It should present eligibility and validity denominators, and conditional and unconditional rejection where a statistical test is used. For rare eligibility, uncertainty must be reported; a small number of accepted windows is not decisive evidence.

Calibration should reproduce the complete evaluation procedure under a defensible null model, including terminal liquidation, selection rules and parameter searches. For a fixed candidate, a conditional null simulation may be appropriate when a sufficiently accurate model is available; for a searched strategy, calibration must reproduce the search and its dependence structure. Model-matched calibration cannot protect against arbitrary misspecification. Calendar-time P&L reporting complements trade-time metrics but is not a universal cure. Where inverse weighting is considered, report the eligibility probability, the support coverage and both the bulk (for example $v_{0.99}^Q$) and the tail ($K$) of the weight distribution before interpreting an adjusted estimate. Missing support requires assumptions or additional observations, not merely larger weights.

In the illustrative example, the count threshold can be useful for operational eligibility but should not be described as a statistical significance guarantee. The same logic may be relevant to fixed-window evaluation of sequential service, production or computational tasks if the iid marked renewal assumptions are appropriate. We have not empirically validated those other applications.

# 6. Discussion and conclusion

This case study demonstrates a practical failure mode: a minimum count designed to guard against small samples may change which outcomes become eligible for inference. The exact conditional mean identity provides a transparent explanation, and the matched-count experiment separates reward–count dependence from ordinary fixed-count sampling effects within the chosen model. The distinction between 47.18% conditional rejection and 4.01% overall rejection in the independent confirmation is central to the applied message. The weighting diagnostics further illustrate why a theoretically observable region may be difficult to use in finite samples.

The work has important limitations. It uses one additive Gaussian null, fixed barriers, no costs, and no serial dependence or genuine positive-reward alternatives. It does not establish power or performance in real markets. The exploratory matched controls cover only seven cells; the fresh confirmation covers three preselected cells under the same model. Extreme ratio estimates and tail-weight moments remain sensitive to rare durations and finite simulation pools. The paper does not propose a new general theory of renewal processes, selective inference or identification; it uses known statistical mechanisms to explain a specific backtesting practice. The independent confirmation addresses reuse of exploratory random paths but does not establish external validity, robustness to alternative return laws, or performance under real trading conditions.

The central recommendation is consequently modest: treat a completed-event count as part of the observation and selection mechanism rather than a certificate of evidence. Reporting both clocks, keeping exclusion denominators and calibrating the actual pipeline can make the reliability of event-based performance reports more transparent.

# Declarations

**Data and code availability.** All empirical results in this draft are from synthetic simulations. The manuscript reports simulation settings and random seeds. The public review repository is https://github.com/salamon-aska/trade-count-selection-bias. The replication archive is not yet complete or independently reproduced; the current repository is a pre-submission review resource.

**Funding and conflicts of interest.** [Author to complete truthfully, including whether the research was carried out independently of the author's employer.]

**Author contributions.** [Author to specify actual human contributions, the list of authors and their roles, and responsibility for the final paper.]

**Generative AI disclosure.** OpenAI ChatGPT and Anthropic Claude provided substantial assistance with research design, literature exploration, mathematical derivations and counterexamples, code development and review, interpretation of numerical outputs, and manuscript drafting and revision. The tools also assisted with academic English editing and preparation of submission materials. Model identifiers and sessions: [author to verify and complete]. Numerical results were obtained by executing explicitly specified stochastic simulation and statistical analysis code developed with substantial generative AI assistance; numerical observations were not directly sampled from language-model responses. Neither AI system is an author. [Before submission the human author must complete and document independent verification of the proofs, code, outputs, references and text; describe the actual verification and confirm compliance with the journal’s policy. Do not claim verification or editorial approval that has not occurred.]

# References

1. Glynn, P. W., and P. Heidelberger. 1990. Bias Properties of Budget Constrained Simulations. *Operations Research* 38(5): 801–814. https://doi.org/10.1287/opre.38.5.801
2. Bao, Y. 2009. Estimation Risk-Adjusted Sharpe Ratio and Fund Performance Ranking under a General Return Distribution. *Journal of Financial Econometrics* 7(2): 152–173. https://doi.org/10.1093/jjfinec/nbn022
3. Bailey, D. H., and M. López de Prado. 2014. The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management* 40(5): 94–107. https://doi.org/10.3905/jpm.2014.40.5.094
4. Khan, S., and E. Tamer. 2010. Irregular Identification, Support Conditions, and Inverse Weight Estimation. *Econometrica* 78(6): 2021–2042. https://doi.org/10.3982/ECTA7372
5. Manski, C. F. 1989. Anatomy of the Selection Problem. *Journal of Human Resources* 24(3): 343–360. https://doi.org/10.2307/145818
6. Sheppert, A. P. 2026. The GT-Score: A Robust Objective Function for Reducing Overfitting in Data-Driven Trading Strategies. *Journal of Risk and Financial Management* 19(1): 60. https://doi.org/10.3390/jrfm19010060
7. Vardi, Y. 1982. Nonparametric Estimation in Renewal Processes. *Annals of Statistics* 10(3): 772–785. https://doi.org/10.1214/aos/1176345870
8. Kaminski, K. M., and A. W. Lo. 2014. When Do Stop-Loss Rules Stop Losses? *Journal of Financial Markets* 18: 234–254. https://doi.org/10.1016/j.finmar.2013.07.001
9. Imbens, G. W., and C. F. Manski. 2004. Confidence Intervals for Partially Identified Parameters. *Econometrica* 72(6): 1845–1857. https://doi.org/10.1111/j.1468-0262.2004.00555.x

10. Wilson, E. B. 1927. Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association* 22(158): 209–212. https://doi.org/10.1080/01621459.1927.10502953
11. Efron, B., and R. J. Tibshirani. 1993. *An Introduction to the Bootstrap*. Chapman & Hall. https://doi.org/10.1201/9780429246593
12. Kish, L. 1965. *Survey Sampling*. New York: John Wiley & Sons.
13. Glynn, P. W., and W. Whitt. 1993. Limit Theorems for Cumulative Processes. *Stochastic Processes and their Applications* 47(2): 299–314. https://doi.org/10.1016/0304-4149(93)90019-Z

# Appendix A. Secondary analyses and negative results

The V0.2 grid-search analysis chooses the largest unannualized Sharpe ratio across 100 candidates. The trade-clock search includes terminal partial trades, unlike the main completed-cycle analysis. The calendar-clock search uses daily P&L. Naive rejection frequencies are 64.10% and 79.45%, respectively. The implemented deflated Sharpe ratio (DSR) with nominal candidate count 100 gives 8.90% and 0.02%; with a participation-ratio effective count it gives 11.64% and 1.11%. Independent model-matched maximum calibration gives 4.36% and 4.84%. The DSR threshold is not intrinsically a 5% familywise-error rule. The participation ratio is an effective-dimensionality proxy, not necessarily the effective number of maxima trials. Complete-case correlation calibration in V0.3 retained only 650 paths at minimum count five and none at ten or higher. These results are secondary diagnostics, not a claim against the general validity of DSR.

A first-order joint renewal CLT (see the cumulative-process framework in [13]) suggests a residualization using the count statistic. With $\mu_L=E[L]$, $\sigma_L^2=\operatorname{Var}(L)$, $\rho=\operatorname{Corr}(L,X)$, define $W_T=(N_T-T/\mu_L)/(\sigma_L\sqrt{T/\mu_L^3})$ and $Z_T=\sqrt{N_T}\bar X/s$. Under the zero-mean iid cycle model with finite second moments and positive variances, their joint limit has standard normal marginals and correlation $-\rho$. For a fixed standardized count threshold and $|\rho|<1$, the residual $(Z_T+\rho W_T)/\sqrt{1-\rho^2}$ has a standard normal conditional limit when parameters are known. This is an asymptotic control-variate construction; it does not imply moment convergence or validity at increasingly rare selection thresholds. In the exploratory finite-window comparison using a normal critical value, the 0.5/10 cell at m=30 rejects at 47.68% before and 20.69% after residualization; at 0.5/8, m=2, the rate worsens from 14.59% to 17.07%. We do not recommend this as a validated finite-sample correction.


# Appendix B. Proofs and integrability

**Proposition 1.** Proof. For every $n\ge m$, the event $\{N_T=n\}=\{S_n\le T<S_n+L_{n+1}\}$ is invariant under permutations of the first $n$ pairs. Thus the expectation of their average multiplied by the event indicator equals the expectation of the first term multiplied by that indicator. Summing over $n\ge m$ yields $E[g(X_1,L_1);A_m]$. Applying the conditional law above and dividing by $P(A_m)$ proves the identity. For absolute integrability, first apply the argument to $|g|$ using Tonelli's theorem. The case $m=1$ is the fixed-budget sample-mean identity discussed in [1]; the extension to $m\ge2$ is immediate.

For the increment identity, $Q_{m+1}$ is absolutely continuous with respect to $Q_m$ because $w_{m+1}\le w_m$. Its Radon–Nikodym derivative is $r_m/E_{Q_m}[r_m]$. Subtracting the two expectations of $h(L)$ gives the stated covariance identity. The conditional empirical moments need not imply moments of a ratio estimator.

Under the Gaussian null, the complete-trade exit time has a geometric tail bound: from any position inside the finite interval, an increment larger than the full barrier width in absolute value in either specified direction has a fixed positive probability of causing exit. Thus the exit time has finite expectation. The stopped zero-drift signed Gaussian sum is uniformly integrable in $L^2$ by the stopped-martingale second-moment bound, so its expected reward is zero. This justifies the null estimand used here.


# Tables

**Table 1.** One-sided t-test rejection percentages with 95% Wilson intervals [10], 10,000 exploratory windows per cell. TP and SL are in daily innovation standard deviations. "Valid" counts windows that are both eligible and numerically valid; A and the matched control share these windows. Source: V0.3 A_vs_E_matched_N.csv and A_by_min_trades.csv.

| TP/SL | m | Valid | A cond. % (95% CI) | Matched cond. % (95% CI) | A overall % |
|---|---:|---:|---:|---:|---:|
| 0.5/10 | 2 | 9,999 | 17.3 (16.6–18.1) | 18.3 (17.6–19.1) | 17.32 |
| 0.5/10 | 20 | 5,012 | 20.3 (19.2–21.4) | 12.1 (11.2–13.0) | 10.18 |
| 0.5/10 | 30 | 904 | 47.2 (44.0–50.5) | 16.8 (14.5–19.4) | 4.27 |
| 1/10 | 20 | 1,719 | 37.6 (35.3–39.9) | 13.7 (12.2–15.4) | 6.46 |
| 1/10 | 30 | 43 | 60.5 (45.6–73.6) | 7.0 (2.4–18.6) | 0.26 |
| 3/3 | 20 | 2,760 | 4.9 (4.2–5.8) | 5.0 (4.2–5.8) | 1.36 |
| 10/0.5 | 30 | 933 | 0.0 (0.0–0.4) | 1.4 (0.8–2.4) | 0.00 |

**Table 2.** Plug-in conditional mean and simulated window mean, TP/SL=0.5/10. Rewards are in reference-notional units. Source: ensi_exact_vs_sim.csv.

| m | Pred. pass | Obs. pass | Pred. mean | Obs. mean | Window SE |
|---|---:|---:|---:|---:|---:|
| 2 | .99978 | .9999 | −.00002 | −.00000 | .00009 |
| 10 | .94416 | .9439 | .00065 | .00062 | .00008 |
| 20 | .49985 | .5012 | .00323 | .00306 | .00008 |
| 30 | .09099 | .0904 | .00587 | .00573 | .00013 |

**Table 3.** Empirical overlap diagnostics for TP/SL=0.5/10, recomputed from the round-4 pool (2,000,000 complete cycles, identical seed) with full-pool denominators. $v_{0.99}^Q$ is the 99th percentile of $v$ under $Q$ (weights proportional to $w_m$). The K range is over four disjoint 500,000-cycle sub-pools. Source: ENSI_round4b_corrected_diagnostics.csv.

| Renewal horizon | m | Pass probability | Empirical q | K | K sub-pool range | $v_{0.99}^Q$ |
|---:|---:|---:|---:|---:|---:|---:|
| 251 | 10 | .945 | .000083 | .894 | .83–.98 | 1.2 |
| 251 | 20 | .502 | .000175 | 4.5e−5 | 3.0e−5–7.8e−5 | 2.0 |
| 251 | 30 | .092 | .000351 | 1.6e−9 | 1.4e−9–1.9e−9 | 3.0 |
| 501 | 30 | .831 | 0 observed | .993 | .993–.994 | 1.3 |

**Table 4.** Independent confirmation (10,000 windows per cell). A and E denote completed trades and count-matched independent rewards. Joint valid counts equal eligible counts in all displayed rows; denominators of individual rates are shown. Paired difference and 95% percentile bootstrap interval are in percentage points; overall rates have denominator 10,000. The first row is primary; remaining rows are descriptive. Source: confirmation_summary.csv and confirmation_paths.csv.gz.

| TP/SL | m | Eligible | A rejects / valid (%) | E rejects / valid (%) | Paired difference [95% CI] | Overall A/E (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5/10 | 30 | 850 | 401/850 (47.18%) | 105/850 (12.35%) | 34.82 [30.82, 38.82] | 4.01 / 1.05 |
| 0.5/10 | 2 | 9,998 | 1699/9998 (16.99%) | 1883/9998 (18.83%) | -1.84 [-2.91, -0.77] | 16.99 / 18.83 |
| 0.5/10 | 20 | 5,081 | 1050/5081 (20.67%) | 592/5081 (11.65%) | 9.01 [7.60, 10.45] | 10.50 / 5.92 |
| 3/3 | 20 | 2,743 | 132/2743 (4.81%) | 147/2743 (5.36%) | -0.55 [-1.71, 0.62] | 1.32 / 1.47 |
| 10/0.5 | 30 | 965 | 0/965 (0.00%) | 23/965 (2.38%) | -2.38 [-3.42, -1.45] | 0.00 / 0.23 |

# Figure legends

![Figure 1. Exploratory rejection rates, reconstructed from rounded Table 1 values (TP/SL = 0.5/10).](figure1.png){width=85%}

![Figure 2. Eligibility probability and marginal Kish weight ratio, reconstructed from rounded Table 3 values (renewal horizon 251, TP/SL = 0.5/10).](figure2.png){width=85%}