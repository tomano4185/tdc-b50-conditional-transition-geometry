# B50 Conditional Transition-Geometry Specificity — Reproducibility Repository

This repository contains the public source code, frozen machine outputs, publication
tables, figures, and provenance records for the B50 conditional transition-geometry
analysis on prime-gap transition operators.

## Scientific scope

The public result is deliberately narrow. The repository tests whether physically
adjacent prime-gap operator windows exhibit transition-geometry specificity relative
to matched controls, whether that effect survives same-corpus phase offsets, and
which measured transition channels recapture the effect.

The repository is a falsification and reproducibility package. It does not present
a theorem about the primes and does not claim independent arithmetic-interval
replication.

## Repository layout

- `code/` — frozen public B50 and publication-generation source files.
- `data/derived/` — frozen B50.6–B50.12 machine outputs used by the publication.
- `tables/` — publication-facing machine-derived CSV tables.
- `figures/` — publication-facing PNG and vector-PDF figures.
- `docs/` — provenance, reproducibility, scope, claims, and data-availability notes.
- `manifests/` — SHA-256 inventories, stage verdicts, audits, and source bindings.

## Frozen analysis lineage

The baseline transition operator is built from positive consecutive prime gaps using
a 64-bin window-adaptive min-max discretization. The resulting 64 x 64 transition
matrix is flattened to a 4096-dimensional representation before downstream
dimensionality and transition-geometry analyses.

The public lineage includes:

1. B50.1–B50.5 state/representation adequacy and null-oriented audits.
2. B50.6 physical-adjacency specificity.
3. B50.7 external offset replication.
4. B50.8 multi-offset robustness.
5. B50.9 attempted disjoint-corpus replication, retained as a negative/blocked
   transparency artifact.
6. B50.10 conditional transition-channel mechanism audit.
7. B50.11 exhaustive channel-subset/redundancy audit.
8. B50.12 frozen-pair cross-offset replication.

Exact stage verdicts and SHA-256 bindings are preserved under `manifests/`.

## Reproducibility

Start with:

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_REPOSITORY.ps1
```

Then read `docs/REPRODUCIBILITY.md`.

## Citation and declarations

Machine-readable citation metadata are supplied in `CITATION.cff` once the author
metadata freeze is complete. Funding, competing-interest, affiliation, and
correspondence statements are not inferred by the repository builder.

## Licensing

This repository uses a fixed dual-license policy for the public release:
executable source code and scripts are licensed under MIT; non-code public
research materials are licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). See `LICENSE`, `LICENSES/`, and
`docs/LICENSING.md` for the scope map.