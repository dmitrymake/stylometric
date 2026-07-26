# Blind benchmark dual-manifest migration

Status: **required before scientific blind scoring**.

The legacy `1.0` benchmark schema combines public rows and full source
provenance. A blind row therefore either exposes identity-bearing values
(`source.sha256`, revision, path, work/edition metadata) or omits information
needed to prove train/blind isolation. That contract cannot be repaired safely
in place.

Until a versioned replacement is registered:

- public CLI/file scoring of any manifest containing `split=blind` fails closed;
- only the explicitly named `synthetic_integration_only=True` path can exercise
  scoring mathematics, and its result is not eligible for a scientific claim;
- exact source-byte/path reuse is rejected and identity metadata is forbidden
  on public blind rows where the legacy schema permits doing so.

The replacement must contain:

1. a public redacted manifest with opaque document ids and no source/work/
   edition/topic/register identity;
2. a custodian full-provenance manifest with exact bytes, paths, revisions and
   grouping identities;
3. an immutable mapping and digests binding both manifests, truth, protocol,
   code version and escrow timestamp/signature;
4. centralized train/development/test/blind content and group isolation checks.

Existing `1.0` files remain historical/integration artifacts. They must not be
silently upgraded or used for a new headline.
