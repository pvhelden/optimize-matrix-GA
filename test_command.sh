#!/bin/bash

venv/Downloads/bin/python optimize-matrix-GA.py -v 4 \
  -t 20 -g 2 -c 5 -s 3 \
  -m data/matrices/PRDM5_GHTS_YWK_B_AffSeq_B1_PRDM5.C2_peakmo-clust-trimmed.tf \
  -p data/sequences/YWK_B_AffSeq_B1_PRDM5.C2.fasta \
  -n data/sequences/YWK_B_AffSeq_B1_PRDM5.C2_rand-loci.fa \
  -b data/bg_models/equiprobable_1str.tsv \
  -r "docker run -v /Users/jvanheld/no_backup/rsat_github/optimize-matrix-GA:/home/rsat_user \
    -v /Users/jvanheld/no_backup/rsat_github/optimize-matrix-GA/rsat_results:/home/rsat_user/out \
    eeadcsiccompbio/rsat:2024-08-28c rsat" \
  --output_prefix results/optimized_matrices/PRDM5_GHTS_YWK_B_AffSeq_B1_PRDM5.C2_peakmo-clust-trimmed_
