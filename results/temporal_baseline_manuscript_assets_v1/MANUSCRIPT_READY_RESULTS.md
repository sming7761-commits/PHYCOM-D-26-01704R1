# Modern temporal baseline results

We compared PhyGuard against fourteen modern temporal classifiers under the same five stratified repeats, with 648/216/216 train/validation/test sequences per repeat.

PhyGuard achieved selected accuracy 0.9432 ± 0.0136, coverage 0.8319 ± 0.0261, false-specific rate 0.0556 ± 0.0176, physical exact match 0.7847 ± 0.0281, and control abstention 0.9444 ± 0.0176.

MiniRocket, MultiRocket, MultiRocket-Hydra, and TCN had small positive point differences in selected accuracy, but every 95% paired cluster-bootstrap interval crossed zero. Each also increased false-specific risk and reduced control abstention relative to PhyGuard.

- **MINIROCKET**: selected-accuracy difference +0.0101 (95% CI -0.0177 to +0.0379); false-specific-rate difference +0.0778 (95% CI +0.0268 to +0.1312, Holm-adjusted p=0.0260); control-abstention difference -0.0778 (95% CI -0.1312 to -0.0268).
- **MULTIROCKET**: selected-accuracy difference +0.0184 (95% CI -0.0053 to +0.0432); false-specific-rate difference +0.0722 (95% CI +0.0251 to +0.1219, Holm-adjusted p=0.0306); control-abstention difference -0.0722 (95% CI -0.1219 to -0.0251).
- **MULTIROCKET_HYDRA**: selected-accuracy difference +0.0215 (95% CI -0.0030 to +0.0465); false-specific-rate difference +0.0806 (95% CI +0.0307 to +0.1332, Holm-adjusted p=0.0132); control-abstention difference -0.0806 (95% CI -0.1332 to -0.0307).
- **TCN**: selected-accuracy difference +0.0015 (95% CI -0.0275 to +0.0303); false-specific-rate difference +0.0833 (95% CI +0.0369 to +0.1305, Holm-adjusted p=0.0056); control-abstention difference -0.0833 (95% CI -0.1305 to -0.0369).

No temporal baseline Pareto-dominated PhyGuard on either the selected-accuracy/coverage/false-specific-rate triplet or the extended five-metric set including physical exact match and control abstention.

The main manuscript table uses the representative model subset fixed in the pre-execution protocol. Results for all fourteen baselines are retained in the supplementary table.

## Statistical boundary

The paired bootstrap used 10,000 replicates over the 726 unique original-sample clusters appearing across the five repeated test sets. The five-repeat exact sign-flip test has a minimum attainable nonzero two-sided p-value of 0.0625 and is therefore reported as a small-sample robustness check rather than the primary inferential test.
