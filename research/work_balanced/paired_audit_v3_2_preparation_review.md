# Paired-audit v3.2 preparation remediation record

Status: remediation complete and owner-accepted for evaluator implementation. The security review
was terminated by the owner; no independent security-audit PASS or equivalent claim is made. This record
does not approve a freeze, evaluator/evidence adapter, execution preflight, authorization, fit,
prediction, result, headline, or scientific publication. The v3.2 protocol, the literal three
exclusions, the 47/252 · 43/248 LOBO and 22/134 RuAA universes, and the 16/11 scientific matrix are
unchanged.

## Atomic storage contract

`paired_audit.preparation_bundle.v1` stores exactly one digest-named object below
`paired_audit_v3_2_bundles/`. Its bundle-local `corrected_corpus/`, both fold manifests, both audits,
`bundle_inventory_v1.json`, `candidate.json`, and `SHA256SUMS` are assembled below one hidden stage.
The staged tree is reconstructed and exact-verified from the historical parent, RuAA selection,
config, and protocol before publication. The only publication operation is
`renameat2(RENAME_NOREPLACE)` under `.publish.lock`. No fallible scientific gate runs after that
rename. A child outside a complete verified bundle has no published/promoteable status, and no
`current` or candidate pointer is created.

Existing destinations take the read-only `verify_v3_2_candidate` path. Exact recursive inventory,
types, single-link status, modes, sizes, bytes, candidate self-hash/file map, canonical
`SHA256SUMS`, safe bundle-local child path, child manifest, audits, folds, and all cross-bindings are
recomputed. A partial/tampered destination is neither overwritten, deleted, repaired, nor chmodded.

The preparation, candidate verification/resume, and CLI entry path first execute one fail-closed
storage-capability gate, before reading an input or creating an output/lock/stage. The supported
storage contract requires non-zero `O_NOFOLLOW`, `O_DIRECTORY`, `O_CLOEXEC`, and the other open flags
the implementation uses; fd-relative `open`, `mkdir`, `stat`, `unlink`, `rename`, and directory
listing; `fcntl.flock`; and Linux `renameat2(RENAME_NOREPLACE)`. An absent, zero, or unusable
primitive is an unsupported platform and an immediate hard stop. There is no degraded mode.

All v3.2-reachable candidate, manifest, audit, inventory/file-map, `SHA256SUMS`, historical/corrected
corpus, work/source, and CLI hash/manifest reads use pinned directory chains and
`openat(O_NOFOLLOW|O_CLOEXEC)`. Each captured file is a single-link regular file; its type, mode,
size, inode state, and path binding are checked before and after the descriptor read, and JSON parse
and hashing consume the same captured bytes. Candidate schema, canonical JSON, self-hash, basename,
and the two exact literal `corrected_corpus` bindings pass before any child listing, parse, hash, or
descent. Missing output and bundle-parent directories use mkdirat/openat open-or-create semantics:
a concurrent `FileExistsError` is reopened and strictly checked, never treated as publication
failure or repaired.

## Storage threat-model boundary

The application storage contract protects against concurrent unprivileged-writer tampering involving
symlinks, hardlinks, special files, paths, bytes, sizes, modes, partial publication, and competing
creation/publication. The kernel and the process's current stable mount namespace are trusted
computing base. Scientific identity binds bundle-relative paths, types, modes, sizes, and exact bytes;
it deliberately does not bind physical mount, device, inode, or mount-ID origin. A same-byte bind
mount therefore does not change scientific identity.

Privileged mount-namespace manipulation, bind-remount or mount swap, kernel/filesystem compromise,
root or `CAP_SYS_ADMIN` adversaries, and denial of service are outside the application threat model.
The application does not claim to detect hostile bind mounts. Operational mount attestation or a
private execution namespace may be considered later as a separate execution-preflight control; it is
not part of this remediation, the scientific hashes, or this preparation review.

## Identity disposition

The old `a70d82f2` candidate was unapproved. Scientific serialized artifacts remained byte-identical,
so their identities remain stable. The relative layout, candidate file map, inventory, checksums,
and storage contract changed, so the candidate/bundle identity rotated.

| Identity | a70 value | remediated value | Disposition |
|---|---|---|---|
| historical parent | `15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029` | same | stable input |
| historical manifest self-hash | `8d39132b8b7732af0d39112a4884947caa1125e24adae2281ab8f5f6d4287705` | same | stable bytes |
| parent identity catalog | `d7690b5a7774967a71da8a4556165d77cd1a22bcb68562b773bb4e4b0899047c` | same | reconstructed |
| exclusion policy | `0907c9acd93375d74d404fa88d36b0d5c6061a5a4caf398de44afe9392d9e4bc` | same | protocol unchanged |
| corrected identity catalog | `85dfafc859912b83e88e8400565580eb01db9eac0afff51564d9ab4b57f12137` | same | universe/content unchanged |
| basename audit | `791bf2cc31e439aab50d750177c1fe1a8e56829c525ca34db88153b073f57326` | same | audit unchanged |
| content-isolation audit | `a561bbde00a15071e9d5e0805e4e168ea705c4f52b803a167b2b715fcaf45784` | same | audit unchanged |
| corrected content inventory | `1a9a0779e4e578f38664fd974c7ac4565f12fb4992cf29773a34061fddee8531` | same | corpus bytes/modes unchanged |
| corrected corpus manifest | `a2dc0c4a6d3313354295a482466693d513ee8f77b297dfe4feb852104b2af3f7` | same | serialized manifest unchanged |
| LOBO fold | `117b8ec9f51ef8c6359768a232660b156a16e0d124b8450e9c497b39cf4cc658` | same | rebuilt bytes unchanged |
| RuAA fold | `d8428290c1895ea397367fbea0cab72317e5e3116176244c488d1f7c6f2b682b` | same | rebuilt bytes unchanged |
| candidate / bundle | `eca941167f7a6497a7eb071125fd66f16cb6d16f6fd948e3687ae44152e9fcf7` | `ff620b05f20b81c21732014b553aa739a393c74fe344e6d9f2bd8d80996cef21` | rotated: atomic layout + storage version + recursive inventory/map |

The new exact-inventory self-hash is
`bd54023bc6b9e18bb033f4c7c38c16c7f315c8db2ad8b918cd6ab6e01e25a438`.

## Parity discrepancy and canonical domain

The two old digests describe the same two pristine `a70` output trees and the same 23,771
descendants; they are different hash domains, not differing preparations.

- Reported `ac77504b14b2950d5428614acdfa87a95db99abf0f1a7259ad2c225f59d9b72b`:
  root was the caller-owned output directory (root excluded); namespace
  `paired_audit.preparation_parity.v3_2`; every descendant path was mapped to
  `(file_sha256_or_directory_marker, mode)`. File sizes and explicit file types were absent.
- Review `6cae3398d2d24d79b2c97bb10cdcc35c76a1a42c79dd13196e0698ea9988e13e`:
  the same root and members; namespace `independent.parity.v1`; values additionally contained
  `lstat().st_size` for files and directories. Directory inode size is filesystem-dependent.

The sole new canonical domain is `paired_audit.preparation_bundle_parity.v1`: root is the one
digest-named bundle (root excluded), namespace is bundle-relative POSIX paths, included members are
all descendant directories and regular files, file records bind SHA-256/mode/size, directory records
bind type/mode but deliberately exclude unstable directory inode size, and the mode policy is root
and directories `0755`, files `0644`. Two pristine Git-free preparations and exact resume produced
23,769 included members and digest
`5ef68b662f4868cb99ac4d4c5980b2018bcabdadbedd97b847dfab72625feef6`.

## Fault-injection result

| Injection point | Digest destination | pointer | normal-unwind hidden stage | historical/external target |
|---|---|---|---|---|
| child assembly | absent | absent | removed | unchanged |
| LOBO fold construction | absent | absent | removed | unchanged |
| RuAA fold construction | absent | absent | removed | unchanged |
| RuAA fold verification | absent | absent | removed | unchanged |
| audit write | absent | absent | removed | unchanged |
| candidate write | absent | absent | removed | unchanged |
| `SHA256SUMS` write | absent | absent | removed | unchanged |
| immediately before final rename | absent | absent | removed | unchanged |
| injected final-rename call/failure | absent | absent | removed | unchanged |

Crash-after-success semantics are intentionally different: a successful atomic rename leaves the
complete bundle, and the next invocation pure-verifies and reuses it without staging or mutation.

## Owner disposition and next gate

The owner accepted this preparation for the bounded evaluator/evidence-adapter implementation on
2026-08-10 and terminated the security review. This is an owner-side disposition, not an independent
security review verdict. The next independent gate is the evaluator candidate's single bounded
scientific review. Freeze/production-registry/preflight/authorization/execution/headline/publication
remain respectively unapproved, empty, absent, absent, hard-disabled, not authorized, and not
authorized.
