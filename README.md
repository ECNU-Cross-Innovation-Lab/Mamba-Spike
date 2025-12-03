# Mamba-Spike

> Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing

[![CGI 2024](https://img.shields.io/badge/CGI_2024-Published-success)](https://doi.org/10.1007/978-3-031-82021-2_23)
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--82021--2__23-blue)](https://doi.org/10.1007/978-3-031-82021-2_23)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a PyTorch implementation of the paper **"Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing"** published at CGI 2024.

## Overview

Mamba-Spike is a novel neuromorphic architecture that integrates a spiking front-end with the Mamba backbone for efficient and robust temporal data processing.

**Key Features:**

- **Event-driven processing** through Spiking Neural Networks (SNNs)
- **Selective state spaces** for efficient sequence modeling
- **Linear-time complexity** for processing long temporal sequences
- **Energy-efficient computation** through sparse spike representations

### Architecture

| Component | Description |
|-----------|-------------|
| **Spiking Front-End** | LIF neurons with recurrent connections for event-based encoding |
| **Interface Layer** | Fixed time window accumulation with firing rate normalization |
| **Mamba Backbone** | Selective state space models with linear-time complexity |
| **Classification Head** | Layer normalization with class predictions |

## Setup

```bash
# Create conda environment
conda create -n mambaspike python=3.9
conda activate mambaspike

# Install dependencies
pip install -r requirements.txt
```

## Datasets

The project supports five neuromorphic datasets:

| Dataset | Resolution | Classes | Description |
|---------|------------|---------|-------------|
| N-MNIST | 34x34 | 10 | Neuromorphic MNIST |
| DVS Gesture | 128x128 | 11 | Dynamic hand gestures |
| CIFAR10-DVS | 128x128 | 10 | Neuromorphic CIFAR-10 |
| Sequential MNIST | 28x28 | 10 | Temporal MNIST |
| N-TIDIGITS | 64 channels | 11 | Neuromorphic audio |

**Data Format:** Place preprocessed data in `data/preprocessed/` as pickle files:
- `{dataset}_train.pkl` - Training data
- `{dataset}_test.pkl` - Test data

Each pickle file should contain: `{'data': np.ndarray, 'labels': np.ndarray}`

## Training

```bash
python train.py --dataset <dataset_name> --batch-size <size> --lr <learning_rate> --epochs <num>
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | nmnist | Dataset name |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 200 | Training epochs |

## Evaluation

```bash
python evaluate.py --checkpoint results/<experiment>/checkpoint_best.pth
```

**Options:**
- `--analyze-temporal` - Temporal analysis
- `--compare-paper` - Compare with paper results

## Results

| Dataset | Mamba-Spike | Mamba | SLAYER | DECOLLE |
|---------|:-----------:|:-----:|:------:|:-------:|
| DVS Gesture | **97.8%** | 96.8% | 93.6% | 95.2% |
| TIDIGITS | **99.2%** | 98.7% | 97.5% | 98.3% |
| Sequential MNIST | **99.4%** | 99.3% | - | - |
| CIFAR10-DVS | **92.5%** | 91.8% | 87.3% | 89.6% |

## Project Structure

```
mambaspike/
├── data/
│   ├── __init__.py
│   └── dataset_loader.py      # Data loading utilities
├── models/
│   ├── __init__.py
│   └── mamba_spike.py         # Model architecture
├── architecture/
│   └── README.md              # Architecture documentation
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── requirements.txt           # Dependencies
└── README.md
```

## Citation

```bibtex
@inproceedings{qin2025mambaspike,
  title     = {Mamba-Spike: Enhancing the Mamba Architecture with a Spiking
               Front-End for Efficient Temporal Data Processing},
  author    = {Qin, Jiahao and Liu, Feng},
  booktitle = {Advances in Computer Graphics: 41st Computer Graphics
               International Conference, CGI 2024},
  pages     = {303--315},
  year      = {2025},
  publisher = {Springer},
  series    = {Lecture Notes in Computer Science},
  volume    = {15339},
  doi       = {10.1007/978-3-031-82021-2_23}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the paper authors.
