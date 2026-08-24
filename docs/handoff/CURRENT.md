# Current Handoff

- State: **garbage/legacy campaign complete; topic-validity fixed-8 run authorized and unexecuted**
- Updated: 2026-08-24
- Verified baseline commit: `fc335823`
- Branch/worktree at capture: attached `main`, clean; sixty-two commits ahead of `origin/main`;
  local branches are exactly `main` and `release`.
- Active task: `docs/tasks/2026-08-23-topic-validity-measured-fixed8.md` (Result pending)
- Active standard: `docs/agentic/STANDARD.md` v1.3

This file only routes the next session. Scientific status and authorization remain owned by the
governance ledger, protocol, executable gates, and completed task evidence.

## 1. Verified current state

| Item | Verified status | Evidence owner |
|---|---|---|
| v3.2 preparation/frozen inputs | Owner-accepted evaluator input; identities unchanged | preparation/reconciliation tasks |
| Evaluator receipt | Independent expectations and semantic recomputation implemented | receipt-closure task |
| Topic validity | Synthetic mechanism confirmed; real corrected-LOBO effect still unmeasured | topic-validity tasks |
| Topic-validity execution | Owner-authorized measured fixed-8, review-corrected, verified ready, **not executed** | ledger `topic_validity_execution` |
| Production evaluator | Registry empty; candidate unregistered | `CONFIRMATORY_EVALUATOR_REGISTRY` |
| Freeze/preflight/authorization | `None` / absent / absent | runner + governance ledger |
| Confirmatory execution | Hard-disabled | runner |
| Headline/publication | Not authorized; no site/public bytes changed | governance ledger |
| Repository hygiene | No private objects in the index, in `HEAD` history, or in any other ref/stash | `check_release_hygiene.py --audit-local-refs` |

## 2. Material change after the prior handoff

- A bounded garbage/legacy campaign ran under `docs/tasks/2026-08-24-repo-garbage-and-legacy-campaign.md`:
  eight merged branch refs removed; ≈1.37 GB of ignored duplicates and regenerable caches deleted;
  sixteen orphan campaign runners (3 731 LOC) removed with the inventory recomputed to 287 paths and
  digest `e1f0f146…`; twenty-one p0-anchored dossier artifacts retired from `docs/`.
- Three unrelated-history branches plus both stashes were retired after an external, restore-verified
  backup bundle (`~/backup/stylo-history-20260824/`, sha256 `6859b2de…`). `.git` fell from 431 MB to
  4.1 MB and the local copyright exposure is gone.
- Gates after the campaign: inventory OK, release hygiene OK including the local-ref audit, site
  provenance 93/1 verified with a byte-identical regeneration, full pytest green.

## 3. Next gate

Execute the authorized fixed-8 topic-validity study: rebuild the temporary v3.2 bundle from
`data/audit_corpus/15d265e0…` with `data/ruaa_bench_v1/manifest.json`, run the no-fit preflight, then
one uninterrupted run of 992 fits writing only
`research/evidence/topic_validity_lobo_v1/aggregate.json`. Budget: ~52 min representation warm plus
~26 h of fits, hard stop without output at 30 h. Nothing else may mutate `src/stylo` while it runs —
`execution_source_sha256` binds that tree and fork workers read it for the whole window.

After the aggregate exists, an independent clean-context audit precedes any model-semantics decision.
Registration, freeze, preflight, headline and publication each still require separate authorization.

## 4. Revalidation conditions

Revalidate before mutation if history does not contain `fc335823`, if bound governance/source hashes
differ, if registry/freeze/preflight/authorization status changes, if the worktree contains unknown
WIP, or if a new R3 factor appears.

## 5. References

- Campaign: `docs/tasks/2026-08-24-repo-garbage-and-legacy-campaign.md`
- Run task: `docs/tasks/2026-08-23-topic-validity-measured-fixed8.md`
- Scientific status: `research/governance/status_ledger.json`, `research/ROADMAP.md`
- Protocol: `research/work_balanced/paired_audit_protocol.md`
- Process: `docs/agentic/STANDARD.md`, `docs/handoff/README.md`
