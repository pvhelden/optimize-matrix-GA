## Installation

```
git clone https://github.com/pvhelden/optimize-matrix-GA.git
cd optimize-matrix-GA
```

## Dependencies

### python and libraries

Minimal required python version 

The required python libraries are specified in the file requirements.txt

```
pip install virtualenv # if you don't already have virtualenv installed
virtualenv venv --python=python3.12 # to create your new environment (called 'venv' here)
source venv/bin/activate # to enter the virtual environment
pip install -r requirements.txt # to install the requirements in the current environment
```

### RSAT docker installation

The RSAT suite can be installed in various ways (https://rsa-tools.github.io/installing-RSAT/), but we strongly reecommend the docker container, which simlpifies the installation. 

```docker pull eeadcsiccompbio/rsat:20240820```

**Note**: the  RSAT version should be 20240820 or ulterior. 

### RSAT docker configuration

Docker should be configured to let it access in read/write mode the directory where we will run the analyses. 
Here is an example of configuration. 

```
export RSAT_VERSION=20240820 # use this version or a later one
export BASE_DIR=$PWD # base directory required for docker to access the data files
mkdir -p $BASE_DIR
export RSAT_CMD="docker run -v ${BASE_DIR}:/home/rsat_user -v ${BASE_DIR}/results:/home/rsat_user/out eeadcsiccompbio/rsat:${RSAT_VERSION} rsat"
echo ${RSAT_CMD} # check the format
$RSAT_CMD -h # print rsat help message
```
