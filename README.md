# Cross-lingual Inflection as a Data Augmentation Method for Parsing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2022.insights-1.7/)

This repository contains the official implementation for the paper:
**[Cross-lingual Inflection as a Data Augmentation Method for Parsing](https://aclanthology.org/2022.insights-1.7/)**
*Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares*
Presented at the **Third Workshop on Insights from Negative Results in NLP** (ACL 2022).

---

## Overview

This project proposes a morphology-based data augmentation method for low-resource dependency parsing. The core idea is to train a morphological inflector on a target low-resource language and apply it to a related rich-resource treebank. This creates "cross-lingual inflected" (x-inflected) treebanks that mimic the target language's morphology while retaining the rich-resource syntax.

## Project Structure

The repository is organized as follows:

- `src/`: Core logic for data conversion and morphological translation.
  - `main.py`: Main functions for UD-UM conversion and inflection.
  - `const.py`: Centralized configuration and paths.
- `scripts/`: Entry-point scripts.
  - `train.py`: Script to train a morphological inflector.
  - `x-inflect.py`: Script to generate x-inflected treebanks.
- `external/`: External tools and submodules.
  - `neural-transducer`: Sequence-to-sequence model for morphological inflection.
  - `ud-compatibility`: Scripts for UD to UniMorph conversion.
- `notebooks/`: For analysis and visualization.
- `results/`: Directory for output treebanks and logs.

## Installation

1. **Clone the repository and submodules:**
   ```bash
   git clone --recursive https://github.com/amunozo/x-inflection.git
   cd x-inflection
   ```

2. **Install dependencies:**
   Ensure you have the required dependencies for `neural-transducer` and `ud-compatibility`.

## Usage

### Training a Morphological Inflector
```bash
python scripts/train.py --lang [LANG_CODE] --dir [DATA_DIR]
```

### Generating X-Inflected Treebanks
```bash
python scripts/x-inflect.py --treebank_folder [UD_FOLDER] --model_folder [MODEL_DIR] --output [OUTPUT_DIR]
```

## Citation

```bibtex
@inproceedings{munoz-ortiz-etal-2022-cross,
    title = "Cross-lingual Inflection as a Data Augmentation Method for Parsing",
    author = "Mu{\~n}oz-Ortiz, Alberto and
      G{\\'o}mez-Rodr{\\'i}guez, Carlos and
      Vilares, David",
    editor = "Sedoc, Jo{\~a}o and
      Balasubramanian, Niranjan and
      Goldwasser, Dan and
      Riedel, Sebastian",
    booktitle = "Proceedings of the Third Workshop on Insights from Negative Results in NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.insights-1.7",
    doi = "10.18653/v1/2022.insights-1.7",
    pages = "51--57",
}
```

## Contact

**Alberto Muñoz-Ortiz** - [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Acknowledgments

This work is supported by a 2020 Leonardo Grant for Researchers and Cultural Creators from the FBBVA,3 as well as by the European Research Council (ERC), under the European Union’s Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150). The work is also supported by ERDF/MICINN-AEI (SCANNER-UDC, PID2020-113230RB-C21), by Xunta de Galicia (ED431C 2020/11), and by Centro de Investigación de Galicia “CITIC” which is funded by Xunta de Galicia, Spain and the European Union (ERDF - Galicia 2014–2020 Program), by grant ED431G 2019/01.
