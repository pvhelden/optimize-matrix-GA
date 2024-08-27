# optimize-matrix-GA

Optimize position specific scoring matrices (PSSM) according to their capability to discriminate a positive from a
negative sequence set.

## Requirements

A local instance of the software suite Regulatory Sequence Analysis Tools (RSAT).

See [INSTALL.md](INSTALL.md) for the instlalation instructions.

## Usage example

The local python installation should contain all the required libraries.

```
export PYTHON_PATH=venv/Downloads/bin/python # python path should be adapted to your local settings
$PYTHON_PATH optimize-matrix-GA.py -v 3 -t 10 -g 5 -c 5 -s 5 \
  -m data/matrices/PRDM5_GHTS_YWK_B_AffSeq_B1_PRDM5.C2_peakmo-clust-trimmed.tf \
  -p data/sequences/YWK_B_AffSeq_B1_PRDM5.C2.fasta \
  -n data/sequences/YWK_B_AffSeq_B1_PRDM5.C2_rand-loci.fa \
  -b data/bg_models/equiprobable_1str.tsv \
  -r "${RSAT_CMD}" \
  -o results/optimized_matrices/PRDM5_GHTS_YWK_B_AffSeq_B1_PRDM5.C2_peakmo-clust-trimmed/PRDM5_GHTS_YWK_B_AffSeq_B1_PRDM5.C2_peakmo-clust-trimmed_
```
