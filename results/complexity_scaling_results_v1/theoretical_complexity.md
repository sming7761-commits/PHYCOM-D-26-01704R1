# PhyGuard diagnosis-stage complexity

Let `T` be sequence length, `K` the number of KPI channels, `F_g` the logistic
gate input dimension, `M` the number of resolver trees, and `D` the mean tree
traversal depth.

- Evidence extraction: **O(TK)**
- Logistic gate per interval: **O(F_g)**
- Extra Trees resolver per interval: **O(MD)**
- Total diagnosis time per interval: **O(TK + F_g + MD)**
- Simplified total: **O(TK + MD)**

For the frozen repeat-0 model:

- gate dimension: 19
- resolver dimension: 79
- trees: 200
- total tree nodes: 34730
- mean tree depth: 13.855
- maximum tree depth: 18
- serialized model size: 3417386 bytes

The benchmark is conditioned on a locked candidate interval and excludes
upstream anomaly localization. Absolute runtime is hardware-specific.
