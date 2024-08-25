# optimize-matrix-GA

Optimize position specific scoring matrices (PSSM) according to their capability to discriminate a positive from a negative 
sequence set. 

## Requirements

A local instance of the software suite  Regulatory Sequence Analysis Tools (RSAT). 

See [INSTALL.md](INSTALL.md) for the instlalation instructions. 

## Usage example

The local python installation should contain all the required libraries. 

```
export PYTHON_PATH=venv/Downloads/bin/python # python path should be adapted to your local settings
$PYTHON_PATH optimize-matrix-GA.py -v 3 -t 10 -g 5 -c 5 -s 5 \\
  -m data/matrices/GABPA_CHS_THC_0866_peakmo-clust-trimmed.tf \\
  -p data/sequences/THC_0866.fasta -n data/sequences/THC_0866_rand-loci_noN.fa \\
  -b data/bg_models/equiprobable_1str.tsv -r "${RSAT_CMD}"
```
