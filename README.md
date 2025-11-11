# Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End

[![CGI 2024](https://img.shields.io/badge/CGI_2024-Published-success)](https://doi.org/10.1007/978-3-031-82021-2_23)
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--82021--2__23-blue)](https://doi.org/10.1007/978-3-031-82021-2_23)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a PyTorch implementation of the paper **"Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing"** published in CGI 2024 (Computer Graphics International Conference).

## Overview

Mamba-Spike is a novel neuromorphic architecture that integrates a spiking front-end with the Mamba backbone to achieve efficient and robust temporal data processing. The architecture leverages:

- **Event-driven processing** through Spiking Neural Networks (SNNs)
- **Selective state spaces** for efficient sequence modeling
- **Linear-time complexity** for processing long temporal sequences
- **Energy-efficient computation** through sparse spike representations

### Architecture Components

1. **Spiking Front-End**: Uses Leaky Integrate-and-Fire (LIF) neurons to encode event-based data into sparse spike representations
2. **Interface Layer**: Converts spikes to continuous activations while preserving temporal information
3. **Mamba Backbone**: Processes temporal sequences using selective state space models with linear-time complexity
4. **Classification Head**: Outputs class predictions with layer normalization

## Setup

### 1. Environment Setup

The project uses a conda environment named `mambaspike`:

```bash
conda create -n mambaspike python=3.9
conda activate mambaspike
pip install -r requirements.txt
```

### 2. Dataset Preparation

The project supports three neuromorphic datasets:
- **N-MNIST**: Neuromorphic version of MNIST (34x34 resolution)
- **DVS Gesture**: Dynamic hand gestures (128x128 resolution) 
- **CIFAR10-DVS**: Neuromorphic version of CIFAR-10 (128x128 resolution)

Datasets will be automatically downloaded when running the training script.

## Training

### Basic Training

To train on N-MNIST:
```bash
python train.py --dataset nmnist --epochs 100 --batch-size 32
```

To train on DVS Gesture:
```bash
python train.py --dataset dvsgesture --epochs 150 --batch-size 16 --lr 5e-4
```

To train on CIFAR10-DVS:
```bash
python train.py --dataset cifar10dvs --epochs 200 --batch-size 32 --lr 1e-3
```

### Training Parameters

- `--dataset`: Choose from `nmnist`, `dvsgesture`, `cifar10dvs`
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay for AdamW optimizer (default: 1e-4)
- `--time-window`: Time window in microseconds (default: 300000)
- `--dt`: Time bin in microseconds (default: 1000)
- `--output-dir`: Directory to save outputs (default: ./outputs)

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint outputs/nmnist_*/checkpoint_best.pth
```

With additional analysis:
```bash
python evaluate.py --checkpoint outputs/nmnist_*/checkpoint_best.pth --analyze-temporal --compare-paper
```

## Results

Performance comparison on various neuromorphic datasets:

| Dataset         | Mamba-Spike | Mamba | SLAYER | DECOLLE | Spiking-YOLO |
|-----------------|-------------|-------|---------|---------|--------------|
| DVS Gesture     | **97.8%**   | 96.8% | 93.6%   | 95.2%   | 96.1%        |
| TIDIGITS        | **99.2%**   | 98.7% | 97.5%   | 98.3%   | -            |
| Sequential MNIST| **99.4%**   | 99.3% | -       | -       | -            |
| CIFAR10-DVS     | **92.5%**   | 91.8% | 87.3%   | 89.6%   | 91.2%        |

## Project Structure

```
mambaspike/
├── data/
│   └── dataset_loader.py      # Dataset loading utilities
├── models/
│   └── mamba_spike.py         # Model architecture
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Key Features

1. **Efficient Temporal Processing**: Leverages Mamba's selective state spaces for long sequences
2. **Neuromorphic Front-End**: Native processing of event-based data
3. **Multi-Scale Architecture**: Supports different input resolutions
4. **Flexible Training**: Easy to adapt for different neuromorphic datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{qin2025mambaspike,
  title={Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing},
  author={Qin, Jiahao and Liu, Feng},
  booktitle={Advances in Computer Graphics: 41st Computer Graphics International Conference, CGI 2024},
  pages={303--315},
  year={2025},
  publisher={Springer},
  address={Cham},
  series={Lecture Notes in Computer Science},
  volume={15339},
  doi={10.1007/978-3-031-82021-2_23},
  url={https://doi.org/10.1007/978-3-031-82021-2_23}
}
```

**Reference**: Qin, J., Liu, F. (2025). Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing. In: Magnenat-Thalmann, N., Kim, J., Sheng, B., Deng, Z., Thalmann, D., Li, P. (eds) Advances in Computer Graphics. CGI 2024. Lecture Notes in Computer Science, vol 15339. Springer, Cham. https://doi.org/10.1007/978-3-031-82021-2_23

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the paper authors (see paper for details)

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.