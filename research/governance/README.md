# Governance registries

These machine-readable files replace status prose, comment-only test maps, and ambiguous legacy
ownership:

- `status_ledger.json` is the normative current paired-audit state and binds claims to symbols and
  exact source bytes.
- `requirements.json` maps scientific/control requirements to code symbols and exact pytest
  nodeids. The governance regression suite proves every nodeid still collects.
- `runner_catalog.json` covers every Python entrypoint in `scripts/evaluation` with its claim,
  output, identity, and regression contract.
- `topology.json` records canonical, compatibility, diagnostic, and retired paths plus unique output
  ownership.

Historical narrative remains useful evidence, but it cannot override these current registries.
