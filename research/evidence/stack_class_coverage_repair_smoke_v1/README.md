# Stack-class-coverage repair smoke: reconstructable source evidence

This directory closes the source-provenance gap for the five-fold exploratory
repair smoke. The result artifact deliberately remains outside the public release
tree because it binds private-corpus paths. The exact executable source needed to
audit the run is preserved here instead.

`source_manifest.json` binds:

- the unchanged smoke driver used for the run;
- the four source files at repair commit
  `07a8df82c48b62d88dd65bb262dbb3c4931b0500`;
- the local repair bundle and ignored result artifact by SHA256;
- the corpus, row, fold-manifest and representation-cache identities;
- the five preselected folds and the deliberately excluded inferences.

The preserved files reconstruct their original repository-relative paths below
`source/`. Run `pytest tests/test_repair_smoke_source_reconstruction.py` to verify
every byte, compile the reconstructed Python tree, and—when the ignored result is
locally provisioned—cross-check its self-hash and input bindings.

A passing smoke licenses only a full **exploratory** LOBO evaluation. It does not
support a full-corpus estimate, macro-F1, calibration error, interval, p-value,
confirmatory claim, or headline decision.
