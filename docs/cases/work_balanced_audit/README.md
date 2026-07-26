# Work-balanced centroid adversarial audit

> **WITHDRAWN TARGET VERDICTS:** this is a frozen pre-v2 audit. The `top`
> columns below are historical closed-set diagnostics, not authorship
> attributions. Its passports used iid target-chunk uncertainty and lacked the
> required calibrated open-set applicability gate. Current case-passport v2
> abstains on every target until that gate exists.

Paired rerun under identical features, chunks, labels, seeds, and target texts. Only the train-side author centroid changes.
The registered feasibility threshold is 0.80; a score below it fails even when the permutation p-value is small.

| case | legacy gate | work-balanced gate | delta | legacy p | balanced p | legacy diagnostic top | balanced diagnostic top |
|---|---:|---:|---:|---:|---:|---|---|
| kolokol_herzen_ogaryov | 0.7929 | 0.6643 | -0.1286 | 0.0155 | 0.1109 | herzen_publicistic | ogaryov_wikisource |
| sovremennik_fourway_gate | 0.8875 | 0.9 | 0.0125 | 0.0005 | 0.0005 |  |  |
| chekhonte_budilnik_sredi_milykh | 0.5136 | 0.4321 | -0.0815 | 0.001 | 0.009 | alexander_chekhov | alexander_chekhov |
| taras_bulba_additions_strict_annenkov_binary_fw_2000 | 1.0 | 1.0 | 0.0 | 0.0005 | 0.0005 | gogol | gogol |
| taras_bulba_additions_strict_somov_binary_fw_2000 | 0.925 | 0.925 | 0.0 | 0.0005 | 0.0005 | somov | somov |
| taras_bulba_additions_strict_suspects_v2_fw_2000 | 0.8625 | 0.7958 | -0.0667 | 0.0005 | 0.0005 | gogol | gogol |
| taras_bulba_additions_strict_sameperiod_fw_2000 | 0.8114 | 0.7876 | -0.0238 | 0.0005 | 0.0005 | gogol | gogol |
| taras_bulba_additions_strict_topic_cossack_fw_2000 | 0.8 | 0.7875 | -0.0125 | 0.0005 | 0.0005 | somov | somov |
| taras_bulba_additions_loose_suspects_v2_fw_2000 | 0.8625 | 0.7958 | -0.0667 | 0.0005 | 0.0005 | gogol | gogol |
| taras_bulba_additions_loose_sameperiod_fw_2000 | 0.8114 | 0.7876 | -0.0238 | 0.0005 | 0.0005 | gogol | gogol |
| taras_bulba_additions_loose_topic_cossack_fw_2000 | 0.8 | 0.7875 | -0.0125 | 0.0005 | 0.0005 | somov | somov |
| petersburg_nn_fourway_fw_2000 | 0.881 | 0.8155 | -0.0655 | 0.0005 | 0.0005 | dostoevsky_publicistic | dostoevsky_publicistic |
| petersburg_fd_1847_04_27_fourway_fw_2000 | 0.881 | 0.8155 | -0.0655 | 0.0005 | 0.0005 | sollogub | sollogub |
| petersburg_fd_1847_05_11_fourway_fw_2000 | 0.881 | 0.8155 | -0.0655 | 0.0005 | 0.0005 | dostoevsky_publicistic | dostoevsky_publicistic |
| petersburg_fd_1847_06_01_fourway_fw_2000 | 0.881 | 0.8155 | -0.0655 | 0.0005 | 0.0005 | dostoevsky_publicistic | dostoevsky_publicistic |
| petersburg_fd_1847_06_15_fourway_fw_2000 | 0.881 | 0.8155 | -0.0655 | 0.0005 | 0.0005 | dostoevsky_publicistic | dostoevsky_publicistic |

All target labels above are withdrawn closed-set diagnostics. A feasibility pass
does not establish target applicability; historical passports remain only as
withdrawn legacy artifacts.

The historical bespoke Kolokol panel is audited separately because its corpus and 600-word window differ from the framework spec. Under work-balanced function-word centroids it falls from the historical 0.8667 (p=0.0015) to 0.6857 (p=0.0755), so that feasibility claim is withdrawn. See `custom/kolokol_herzen_ogaryov.work_balanced.json`.

The historical Taras Delta report also used an invalid fixed-prediction null. Its corrected equal-work/full-refit rerun is exploratory: suspects passes in both modes and points to Gogol, but the separable Gogol-Somov binary reverses from Gogol under fixed function words to Somov under learned MFW. The fixed-FW topic panel fails its gate. Cross-feature/panel robustness is therefore absent. See `custom/taras_delta_full_refit_work_balanced.json`.

Supplemental bespoke reruns complete the public limits map. The Sovremennik school axis survives at 1.000 (p=0.0005), while the Chernyshevsky-Dobrolyubov pair falls to 0.700 (p=0.2222). Nekrasov-Panaeva function words fall to 0.650 (p=0.2273), whereas the content-sensitive char-3gram channel remains 0.950 (p=0.0152). See `custom/sovremennik.work_balanced.json` and `custom/nekrasov_panaeva.work_balanced.json`.

Frozen summary/custom artifact hashes are listed in `SHA256SUMS`.
