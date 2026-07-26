"""Confirmatory legacy-versus-work-balanced paired audit control plane.

Purpose-named modules implementing the confirmatory audit of
``research/work_balanced/paired_audit_protocol.md`` (v3.1). Estimator-axis routing and the narrow
stylo LOBO validation live elsewhere; this package owns only the confirmatory control plane:

- :mod:`semantic_parity` — the loader-agnostic semantic-row digest, the frozen legacy anchor, and
  the exact row/byte equality proof (§1.2/§1.3).
- :mod:`corpus` — the audit-only dataset verifier, the immutable audit-corpus builder, the published
  immutable-root loader, and the atomic whole-corpus publication (§1.3/§1.4).
- :mod:`work_subset` — the RuAA nested-panel whole-work subset with three-digest binding (§1.5).

No module here prepares a real corpus, freezes a fold manifest, or runs a confirmatory cell. Those
steps stay gated behind the independent code audit and a separate execution authorization.
"""
