# Case Dossier

## Hypothesis

The large 1842 additions were written by Annenkov, who copied the manuscript under dictation in Rome, 1841.

## Claim

Direct binary comparison against the main documented suspect.

## Results

| target | status | score | gate | p | top | chunks | margin CI |
|---|---|---:|---:|---:|---|---|---|
| strict additions, binary Gogol vs Annenkov | moderate | 96.37 | 1.0 | 0.0005 | gogol | gogol 14/16, annenkov_1840s 2/16 | [0.021665, 0.042208] |
| Путевые записки (Annenkov, held out) | strong | 93.81 | 0.8625 | 0.0005 | annenkov_1840s | annenkov_1840s 16/17, gogol 1/17 | [0.048329, 0.097073] |
| Шинель (Gogol, held out) | strong | 91.56 | 0.8569 | 0.0005 | gogol | gogol 13/14, annenkov_1840s 1/14 | [0.032893, 0.042289] |
| Prokopovich letters to Shevyryov, 1843 | moderate | 91.10 | 1.0 | 0.0005 | gogol | gogol 1/1 |  |
| strict additions vs suspects v2 | moderate | 90.19 | 0.8625 | 0.0005 | gogol | gogol 14/16, annenkov_1840s 2/16 | [0.021665, 0.034852] |
| loose additions vs suspects v2 | moderate | 89.87 | 0.8625 | 0.0005 | gogol | gogol 14/17, annenkov_1840s 3/17 | [0.016838, 0.034781] |
| Тарас Бульба, редакция 1835 (base text) | strong | 89.76 | 0.8625 | 0.0005 | gogol | gogol 13/14, annenkov_1840s 1/14 | [0.023925, 0.035239] |
| strict additions, binary Gogol vs Somov | moderate | 88.53 | 0.925 | 0.0005 | somov | somov 12/16, gogol 4/16 | [0.003628, 0.015046] |
| strict additions | moderate | 88.39 | 0.8625 | 0.0005 | gogol | gogol 14/16, pushkin 2/16 | [0.014007, 0.029963] |
| Шинель (Gogol, held out) | fail | 88.35 | 0.7856 | 0.0005 | not interpreted |  |  |
| loose additions | moderate | 88.31 | 0.8625 | 0.0005 | gogol | gogol 15/17, pushkin 2/17 | [0.012841, 0.030624] |
| Путевые записки (Annenkov, held out) | fail | 87.70 | 0.79 | 0.0005 | not interpreted |  |  |
| Тарас Бульба, редакция 1835 (base text) | fail | 86.50 | 0.79 | 0.0005 | not interpreted |  |  |
| Отцы и дети (Turgenev, held out) | moderate | 86.34 | 0.8568 | 0.0005 | turgenev | turgenev 23/35, dostoevsky 7/35, gogol 3/35, annenkov_1840s 2/35 | [0.008231, 0.018747] |
| strict additions vs suspects panel | fail | 85.13 | 0.79 | 0.0005 | not interpreted |  |  |
| loose additions vs suspects panel | fail | 85.04 | 0.79 | 0.0005 | not interpreted |  |  |
| strict additions vs same-period panel | moderate | 84.84 | 0.8114 | 0.0005 | gogol | gogol 11/16, pushkin 2/16, grebenka 2/16, annenkov_1840s 1/16 | [0.010636, 0.02068] |
| Шинель на казачьей панели | fail | 84.73 | 0.7472 | 0.0005 | not interpreted |  |  |
| loose additions vs same-period panel | moderate | 84.60 | 0.8114 | 0.0005 | gogol | gogol 13/17, annenkov_1840s 2/17, grebenka 1/17, pushkin 1/17 | [0.008168, 0.02134] |
| speech on товарищество · focused diagnostic | moderate | 84.29 | 0.8625 | 0.0005 | gogol | gogol 1/1 |  |
| Тарас Бульба, редакция 1835 (base text) на казачьей панели | moderate | 83.78 | 0.8 | 0.0005 | gogol | gogol 12/14, grebenka 1/14, somov 1/14 | [0.007394, 0.020041] |
| loose additions vs cossack topic panel | moderate | 83.17 | 0.8 | 0.0005 | somov | somov 14/17, gogol 2/17, grebenka 1/17 | [0.003593, 0.018027] |
| strict additions vs cossack topic panel | moderate | 82.90 | 0.8 | 0.0005 | somov | somov 12/16, gogol 4/16 | [0.003628, 0.015046] |
| strict additions · extended controls | fail | 79.25 | 0.7515 | 0.0005 | not interpreted |  |  |
| loose additions · extended controls | fail | 79.08 | 0.7515 | 0.0005 | not interpreted |  |  |
| Дубровский (Pushkin, held out) | fail | 77.95 | 0.73 | 0.0005 | not interpreted |  |  |
| strict additions vs Gogol period pseudo-candidates | fail | 59.39 | 0.5833 | 0.041 | not interpreted |  |  |

## Limitations

- Applies only to large added passages, not to local editorial correction, lexical normalization, typography, or small insertions.
- Applies only to this extracted passage, not to all 1842 additions.
- Control run: expected answer is known; failure invalidates the panel, success does not itself prove the main claim.
- Diagnostic only: epistolary register differs from narrative prose; no attribution claim is made.
- Diagnostic: pseudo-candidates are period slices of one author; невский_проспект and записки_сумасшедшего (1835, «Арабески») excluded from both slices.
- Raw target texts are local research inputs under ignored input_cases/; tracked artifacts publish hashes, specs, passports, and reports.
- Single short passage: diagnostic only, never a strong verdict in this protocol.
- Single short text; single-chunk targets cannot receive a strong verdict.
- Single-chunk targets are diagnostic only and cannot receive a strong verdict.
- Stress test only: the larger panel adds less balanced same-century controls and is not the headline protocol.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_annenkov_binary_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_annenkov_binary_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_annenkov_holdout_v2_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_annenkov_holdout_v2_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_shinel_holdout_v2_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_shinel_holdout_v2_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_diag_prokopovich_letters_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_diag_prokopovich_letters_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_suspects_v2_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_suspects_v2_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_loose_suspects_v2_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_loose_suspects_v2_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_gogol1835_base_v2_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_gogol1835_base_v2_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_somov_binary_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_somov_binary_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_shinel_holdout_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_shinel_holdout_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_loose_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_loose_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_annenkov_holdout_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_annenkov_holdout_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_gogol1835_base_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_gogol1835_base_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_turgenev_holdout_v2_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_turgenev_holdout_v2_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_suspects_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_suspects_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_loose_suspects_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_loose_suspects_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_sameperiod_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_sameperiod_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_shinel_topic_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_shinel_topic_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_loose_sameperiod_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_loose_sameperiod_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_tovarishchestvo_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_tovarishchestvo_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_gogol1835_base_topic_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_gogol1835_base_topic_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_loose_topic_cossack_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_loose_topic_cossack_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_topic_cossack_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_topic_cossack_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_extended_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_extended_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_loose_extended_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_loose_extended_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_control_pushkin_holdout_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_control_pushkin_holdout_fw_2000.passport.json
```

```bash
PYTHONPATH=src .venv/bin/python -m stylo.cli case run docs/cases/taras_hardened/specs/taras_bulba_additions_strict_period_fw_2000.yaml --out docs/cases/taras_hardened/passports/taras_bulba_additions_strict_period_fw_2000.passport.json
```

## Passport Verdicts

- **taras_bulba_additions_strict_annenkov_binary_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.875, margin=0.031875).
- **taras_control_annenkov_holdout_v2_fw_2000**: Gate пройден; target устойчиво ближе к annenkov_1840s (winner_share=0.941, margin=0.071647).
- **taras_control_shinel_holdout_v2_fw_2000**: Gate пройден; target устойчиво ближе к gogol (winner_share=0.929, margin=0.039989).
- **taras_bulba_diag_prokopovich_letters_fw_2000**: Gate пройден; target из одного chunk диагностически ближе к gogol, но single-chunk target не может получить strong verdict (winner_share=1.0, margin=0.030512).
- **taras_bulba_additions_strict_suspects_v2_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.875, margin=0.031875).
- **taras_bulba_additions_loose_suspects_v2_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.824, margin=0.030295).
- **taras_control_gogol1835_base_v2_fw_2000**: Gate пройден; target устойчиво ближе к gogol (winner_share=0.929, margin=0.029743).
- **taras_bulba_additions_strict_somov_binary_fw_2000**: Gate пройден; target чаще ближе к somov, но запас/бутстрап недостаточны для strong verdict (winner_share=0.75, margin=0.009507).
- **taras_bulba_additions_strict_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.875, margin=0.022902).
- **taras_control_shinel_holdout_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.7856, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_bulba_additions_loose_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.882, margin=0.022471).
- **taras_control_annenkov_holdout_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.79, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_control_gogol1835_base_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.79, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_control_turgenev_holdout_v2_fw_2000**: Gate пройден; target чаще ближе к turgenev, но запас/бутстрап недостаточны для strong verdict (winner_share=0.657, margin=0.013941).
- **taras_bulba_additions_strict_suspects_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.79, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_bulba_additions_loose_suspects_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.79, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_bulba_additions_strict_sameperiod_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.688, margin=0.016618).
- **taras_control_shinel_topic_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.7472, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_bulba_additions_loose_sameperiod_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.765, margin=0.015427).
- **taras_bulba_tovarishchestvo_fw_2000**: Gate пройден; target из одного chunk диагностически ближе к gogol, но single-chunk target не может получить strong verdict (winner_share=1.0, margin=0.027398).
- **taras_control_gogol1835_base_topic_fw_2000**: Gate пройден; target чаще ближе к gogol, но запас/бутстрап недостаточны для strong verdict (winner_share=0.857, margin=0.013879).
- **taras_bulba_additions_loose_topic_cossack_fw_2000**: Gate пройден; target чаще ближе к somov, но запас/бутстрап недостаточны для strong verdict (winner_share=0.824, margin=0.010843).
- **taras_bulba_additions_strict_topic_cossack_fw_2000**: Gate пройден; target чаще ближе к somov, но запас/бутстрап недостаточны для strong verdict (winner_share=0.75, margin=0.009507).
- **taras_bulba_additions_strict_extended_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.7515, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_bulba_additions_loose_extended_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.7515, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_control_pushkin_holdout_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.73, permutation_p=0.0005. Атрибуцию давать нельзя.
- **taras_bulba_additions_strict_period_fw_2000**: Feasibility gate не пройден: work_macro_recall=0.5833, permutation_p=0.041. Атрибуцию давать нельзя.
