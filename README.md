[![stars-badge](https://img.shields.io/github/stars/gao-lab/PASSAGE-dev?logo=GitHub&color=yellow)](https://github.com/gao-lab/PASSAGE-dev/stargazers)
[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# PASSAGE
Learning phenotype associated signature in spatial transcriptomics with PASSAGE
![Model architecture](./model.png)

## File Structure
```
PASSAGE
├─data                         # Dataset collection
├─docs                         # Documentation files
│  ├─api
│  ├─tutorials                 # jupyter notebook tutorials
│  ├─_static
│  └─_templates
└─PASSAGE                      # Main Python package
    ├─model
    └─viz
```

## Environment
```
mamba env create -f passage_environment.yaml
mamba activate passage
```

## Installation
```
git clone https://github.com/gao-lab/PASSAGE
cd PASSAGE
python setup.py build
python setup.py install
```

## Citations
```

```
