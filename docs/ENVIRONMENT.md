# Environment

This repository preserves the frozen analysis lineage and publication outputs.

## Runtime captured during repository freeze

Python runtime:

Python 3.14.2

Top-level third-party packages are inferred from imports in `code/*.py`.
When their installed versions can be resolved on the freeze machine,
`requirements.txt` pins those versions. The detailed detection result is stored in
`manifests/DEPENDENCY_AUDIT.csv`.

The exact frozen scientific outputs remain authoritative even if a later runtime
produces small floating-point differences during an independent rebuild.