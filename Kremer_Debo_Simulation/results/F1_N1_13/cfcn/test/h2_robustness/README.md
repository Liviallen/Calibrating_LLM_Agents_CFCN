# H1/H2 follow-up analysis: F1_N1_13/cfcn

GPT validation data, subject_type == 0. The manifest identifies gpt5.4; GPT-4o is a legacy label in the analysis scripts. Inference conditions on fixed trained prompt weights and this run. Validation conversations are independently initialized by cohort/subject; the four sequential decisions within a cohort are dependent. Resampling and clustered inference use cohort_id, not individual rows.

## Choice distribution

| Wait | Q00 purchases/choices | Q05 purchases/choices | Q50 purchases/choices |
|---|---|---|---|
| 1 | 208/208 (100%) | 354/357 (99.16%) | 318/324 (98.15%) |
| 2 | 0/624 (0%) | 4/998 (0.40%) | 26/437 (5.95%) |
| 3 | no observations | 0/29 (0%) | 8/113 (7.08%) |
| 4 | no observations | no observations | 0/25 (0%) |

## Original report diagnostics

Q05/Q00 primary probit failed for all optimizers and aborted before writing the summary CSV or H2 results. Q50/Q00 H1 has missing standard errors. Q50/Q00 H2 reports p=0 with a divergent Q00 slope: Q00 is perfectly separated by waits 1 and 2. Ordinary probit Wald p-values are not reliable in this design.

## H1: lower purchasing at wait 1

OLS purchase ~ treatment at wait=1, cohort-clustered covariance with finite-sample correction and t(G-1) reference. Estimates are probability differences, treatment minus Q00.

| Comparison | Difference, pp | 95% clustered CI, pp | Two-sided p | Holm p within H1 |
|---|---:|---|---:|---:|
| Q05/Q00 | -0.8403 | [-1.7830, 0.1023] | 0.080482 | 0.080482 |
| Q50/Q00 | -1.8519 | [-3.3077, -0.3960] | 0.012763 | 0.025525 |

H1 is not established for Q05. Q50 has support under this asymptotic clustered test, but only six refusals occur, so describe this as tentative/method-dependent rather than a robust confirmation.

Additional mean-bias-reduced probit at wait=1 with cohort bootstrap gives Q05 coefficient -0.510657, percentile CI [-0.808949,0.162157], and Q50 -0.793825, CI [-1.071872,-0.422579]. All 1999 resamples succeeded. The centered non-studentized bootstrap p-value in h1_probit_results.csv is not primary evidence: the sparse asymmetric Q05 distribution makes it disagree with the percentile interval. Do not report its p=0.0465 as confirmation of Q05 H1.

A deliberately different cohort-level sensitivity endpoint, at least one uninformed refusal at wait=1, yields 0/208 Q00 cohorts versus 3/354 Q05 (two-sided Fisher p=0.299370) and 6/320 Q50 (p=0.086036). This respects cohort independence but tests an event probability rather than the original per-choice probability; it is a robustness diagnostic, not an identical H1 test. It underlines that Q50 H1 significance is method-dependent with so few refusal events.

## H2 method 1: mean-bias-reduced probit

P(order)=Phi(alpha+beta*wait+gamma*treat+delta*wait*treat). AS_mean estimation through brglm2, all observed waits. Positive delta corresponds to a flatter latent-index slope. Bootstrap entire cohorts within treatment and original validation batch; seed 20261024, 1999 resamples each, all converged. Grouped binomial fits independently checked against individual-row fits within 1e-5.

| Comparison | delta | 95% percentile bootstrap CI | Approximate two-sided centered-bootstrap p | Holm p within H2 |
|---|---:|---|---:|---:|
| Q05/Q00 | 1.074488 | [0.315237,1.441667] | 0.0020 | 0.0020 |
| Q50/Q00 | 3.435496 | [2.866229,3.809812] | 0.0005 | 0.0010 |

Bootstrap p=(1+sum(abs(delta_boot-delta_hat)>=abs(delta_hat)))/(B+1); approximate, non-studentized and at a reporting floor of 0.0005. CI evidence is primary. The deterministic Q00 pattern means bias reduction supplies finite estimates; resampling cannot create unseen outcome variation. Magnitudes depend on the model and regularization and are not percentage-point changes.

## H2 method 2: probability decline from wait 1 to wait 2

OLS purchase ~ wait2 + treat + wait2*treat, restricted to waits 1 and 2, with cohort-clustered t inference. Interaction D=[p_Q(2)-p_Q(1)]-[p_00(2)-p_00(1)]. OLS estimates verified against direct cell means.

| Comparison | D, pp | 95% clustered CI, pp | Two-sided p | Holm p |
|---|---:|---|---:|---:|
| Q05/Q00 | 1.240335 | [0.2227,2.2595] | 0.017001 | 0.017001 |
| Q50/Q00 | 7.801509 | [5.0922,10.5108] | 2.450559e-8 | 4.901118e-8 |

Both comparisons support H2 under both follow-up methods. Method 2 tests a narrower probability-scale estimand and is not identical to the global probit slope test. Q50's raw rates rise slightly from wait 2 to 3; the overall fitted slope does not imply strict empirical monotonicity at every wait.

## Conclusion

Q05: H1 not statistically established; H2 supported by both methods.
Q50: H1 has tentative, method-dependent support from clustered per-choice tests, with weak exact cohort-event sensitivity; H2 supported by both methods.

These are post hoc follow-up analyses, not proof, and not randomized causal effects of waiting time. Results do not generalize automatically across training runs or prompt specifications. Source scripts, model outputs, and bootstrap draws are saved alongside this report. R uses brglm2 installed temporarily in /tmp/h2-r-library; sessionInfo is recorded.
