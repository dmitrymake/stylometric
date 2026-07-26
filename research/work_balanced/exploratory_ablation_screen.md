# Exploratory screen of work weighting and feature fitting

Status: **completed exploratory evidence**. This screen selected mechanisms for the full per-book
validation; it is neither the confirmatory paired audit nor an external replication.

## Question and fixed design

The screen asked whether work weighting (`W`), work-level feature fitting (`F`), relative
function-word frequencies (`R`), or their complete combination changed held-out-work attribution
enough to justify a substantially more expensive true LOBO run.

- Dataset binding: `docs/screening_panel_v1.json`.
- Panel self-hash: `e881d4905652ff613e89670c204ab4dbb5eb8915fe35aa1249879ee3d13ba0bb`.
- Universe: 43 tested authors and 251 works.
- Split: the same five frozen work-level folds for every model and cell, sized
  `49/48/52/51/51`.
- Seed: `42`.
- Inference: paired author-clustered accuracy bootstrap, 1,000 iterations, 95% interval.
- No work crosses train/test; learned state is fit only on each training fold; chunk probabilities
  are class-aligned and averaged exactly once per held-out work.

The stable cell meanings are:

- `A0`: legacy weighting and feature state;
- `A1`: work weights only;
- `A2`: work-level feature fitting only;
- `A3`: relative function-word frequencies only;
- `A4`: the complete registered work-balanced path.

Unsupported or equivalent cells were recorded explicitly rather than populated with copied metrics.

## Results

Accuracy differences and intervals are paired against the corresponding model's `A0`.

| Model | Cell | Accuracy | Macro-F1 | Δ accuracy | 95% clustered interval |
|---|---:|---:|---:|---:|---:|
| stylo | A0 | 0.87649 | 0.84568 | 0 | [0, 0] |
| stylo | A1 | 0.90438 | 0.87641 | +0.02789 | [+0.00433, +0.05834] |
| stylo | A2 | 0.87649 | 0.84507 | 0 | [-0.01778, +0.01556] |
| stylo | A3 | 0.88446 | 0.85729 | +0.00797 | [-0.00841, +0.02480] |
| stylo | A4 | 0.89641 | 0.86700 | +0.01992 | [-0.00752, +0.05140] |
| bow_lr | A0 | 0.78486 | 0.71079 | 0 | [0, 0] |
| bow_lr | A1 | 0.78884 | 0.75533 | +0.00398 | [-0.04015, +0.05432] |
| bow_lr | A2 | 0.78088 | 0.70574 | -0.00398 | [-0.02381, +0.01128] |
| bow_lr | A4 | 0.76096 | 0.72536 | -0.02390 | [-0.06589, +0.02551] |
| delta_cos:500 | A0 | 0.82072 | 0.79203 | 0 | [0, 0] |
| delta_cos:500 | A2 | 0.82470 | 0.78347 | +0.00398 | [-0.01747, +0.02501] |
| delta_cos:500 | A3 | 0.80876 | 0.76491 | -0.01195 | [-0.03181, 0] |
| delta_cos:500 | A4 | 0.80876 | 0.75922 | -0.01195 | [-0.03384, +0.00820] |
| char_cos | A0 | 0.70120 | 0.65269 | 0 | [0, 0] |
| char_cos | A4 | 0.67331 | 0.62948 | -0.02789 | [-0.08333, +0.02465] |
| majority | A0 | 0.03984 | 0.00178 | 0 | [0, 0] |

For `stylo`, A1 produced `227/251` correct versus `220/251` for A0: seven gains and no losses.
Its five fold deltas were not driven by a single fold, its leave-one-author-out delta remained
positive, and the exploratory author-clustered interval was entirely above zero.

## Interpretation boundary

The registered triage label is `promising_directional_signal`, specifically for the stylo
weights-only mechanism. This justified true 47-class per-book LOBO for A0/A4/A1. It did not select a
new headline, validate the proxy accuracy as a target result, or authorize claims about authorship.
Other model families did not show the same consistent benefit, so the signal must not be generalized
across the model grid.

The ignored strict-JSON evidence is
`docs/exploratory/work_balanced/b4_wfr_pilot_v1.json`, self-hash
`b3623978cea91a00028a84675a6e9687b4075efe2bceebf8c29402542ec87c31`. The implementation was signed
in commit `2f6c3dc3`; historical execution instructions are local-only and non-normative.
