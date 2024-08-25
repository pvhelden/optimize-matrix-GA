# optimize-matrix-GA

Optimize position specific scoring matrices (PSSM) according to their capability to discriminate a positive from a negative 
sequence set. 

## Requirements

A local instance of the software suite  Regulatory Sequence Analysis Tools (RSAT). 

### RSAT docker installation

The RSAT suite can be installed in various ways (https://rsa-tools.github.io/installing-RSAT/), but we strongly reecommend the docker container, which simlpifies the installation. 
The  RSAT version should be 20240820 or ulterior. 

```docker pull eeadcsiccompbio/rsat:20240820```

### RSAT docker configuration

```
export RSAT_VERSION=20240820 # use this version or a later one
export BASE_DIR=$PWD # base directory required for docker to access the data files
mkdir -p $BASE_DIR
export RSAT_CMD="docker run -v ${BASE_DIR}:/home/rsat_user -v ${BASE_DIR}/results:/home/rsat_user/out eeadcsiccompbio/rsat:${RSAT_VERSION} rsat"
echo ${RSAT_CMD} # check the format
$RSAT_CMD -h # print rsat help message
```

## Usage example


```
export PYTHON_PATH=venv/Downloads/bin/python # python path should be adapted to your local settings
$PYTHON_PATH optimize-matrix-GA.py -v 3 -t 10 -g 5 -c 5 -s 5 \\
  -m data/matrices/GABPA_CHS_THC_0866_peakmo-clust-trimmed.tf \\
  -p data/sequences/THC_0866.fasta -n data/sequences/THC_0866_rand-loci_noN.fa \\
  -b data/bg_models/equiprobable_1str.tsv -r "${RSAT_CMD}"
```
