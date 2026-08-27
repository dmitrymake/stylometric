# Fixed-8 topic-validity execution

- Status: superseded (superseded — orchestration replaced by the resumable runner (43f1fdc))
- Owner: Dmitry Purtov
- Baseline: `5e581f850b166c3abef3fdcb2e511a298ddbda5a`
- Type / Risk: implementation + research / R3b
- Approval: inherited exact owner authorization for LOBO-248 A0/A4 current/topic_strict aggregate-only study.

## Goal

Replace only the sequential orchestration with a fixed eight-process fork pool after representation
warm-up, prove byte-identical synthetic aggregate versus serial execution, then complete the approved
992 fits and retain the same single aggregate schema.

## Result

- Candidate `a3d33789`; deadline correction `b67372e0` after one review FAIL.
- Synthetic real-fork serial/parallel records and aggregate bytes matched; worker failure and absolute
  16-hour timeout terminate before output.
- Full run reached `10/992` at 957.7 s (26.4 h projection), so it was stopped with exit 130 under the
  load contract. Aggregate path remained absent.
- Observed fixed-8 memory was 40 GiB used / 21 GiB available on a 62 GiB host; workers added roughly
  8 GiB over baseline. Fixed-16 projects ~13.2 h with ~13 GiB remaining headroom.
- Next task changes only the bound worker constant from 8 to 16; all scientific/output contracts stay frozen.
