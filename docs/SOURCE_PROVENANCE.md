# Source provenance

## Corpus contract

The numerical source corpus is segmented into files corresponding to prime values
in consecutive intervals

`[(i-1) * 10^7, i * 10^7 - 1]`

for segment index `i`. The first historical segment prepends a symbolic `1`; the
operator construction itself uses positive consecutive prime gaps.

The complete raw corpus is not redistributed in this GitHub repository. Public
reproducibility therefore distinguishes frozen-output reproduction from a full
raw-corpus rebuild.

## Operator contract

For each analysis window:

1. load prime values;
2. compute consecutive differences;
3. retain positive gaps;
4. define 64 linearly spaced bins between that window's minimum and maximum gap;
5. digitize successive gaps;
6. count directed bin-to-bin transitions;
7. row-normalize the count matrix;
8. for an empty source row, use that same window's destination marginal;
9. flatten the 64 x 64 matrix to 4096 dimensions.

Because the binning is window-adaptive, the same bin index need not represent the
same absolute gap interval in different windows.

The frozen baseline implementation is source-bound by SHA-256 in the source
inventory and manifests.

## Derived-result lineage

Frozen B50 result directories are copied byte-for-byte into `data/derived/`.
Publication-facing tables and figures are copied from the frozen PUB-003 asset build.
Every exported file is SHA-256 verified.

Some frozen CSV/JSON audit artifacts retain historical absolute workstation or
corpus paths as provenance fields. These strings are preserved byte-for-byte and
are not runtime instructions. They are reported separately in
`manifests/FROZEN_PROVENANCE_PATH_AUDIT.csv`; publication-facing code and
documentation must remain free of machine-specific absolute paths.

No control matching, channel selection, statistical threshold, or scientific
inference is recomputed by the repository export stages.

## Public scope boundary

Only material required to audit and reproduce the public B50 result is included.
Unrelated internal research is outside the repository scope.