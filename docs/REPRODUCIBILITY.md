# Reproducibility

The repository supports three distinct reproducibility levels. They should not be
conflated.

## Level 0 — byte-level repository verification

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_REPOSITORY.ps1
```

This checks the SHA-256 inventory created at repository freeze.

## Level 1 — publication-output audit

The frozen machine outputs are under `data/derived/`; the final publication tables
are under `tables/`; final figures are under `figures/`.

Use the frozen CSV/JSON outputs to independently verify every reported number before
attempting any expensive corpus rebuild. `publication_numbers.json` is the
publication-facing numerical binding.

## Level 2 — analysis-code replay

The public B50 source lineage is under `code/`. `requirements.txt` records the
top-level Python dependencies detected during repository freeze.

Replay must preserve the frozen scientific protocol. In particular, do not alter:

- the physical window sequence;
- the matched-control construction;
- channel definitions;
- subset-selection criteria;
- offset phases;
- significance thresholds;
- frozen pair definition.

Any replay that changes these items is a new experiment, not a reproduction.

## Level 3 — full raw-corpus rebuild

The full prime corpus is not bundled in this repository. A raw rebuild therefore
requires an independently reconstructed corpus satisfying the segmentation contract
in `docs/SOURCE_PROVENANCE.md`.

This level is intentionally separated from publication-output reproduction because
the raw corpus is much larger than the public repository.

## Expected scientific endpoints

The repository preserves the frozen verdicts:

- `PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED`
- `EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED`
- `MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE`
- `INSUFFICIENT_DISJOINT_CORPUS`
- `CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION`
- `PAIRWISE_CORE_RECAPTURES_FULL_EFFECT`
- `PAIR_CORE_CROSS_OFFSET_REPLICATED`

A reproduction should report deviations rather than silently retune the protocol.