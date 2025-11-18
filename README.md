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

1. **Spiking Front-End**: Uses Leaky Integrate-and-Fire (LIF) neurons with recurrent connections to encode event-based data into sparse spike representations
2. **Interface Layer**: Converts spikes to continuous activations using fixed time window accumulation and firing rate normalization
3. **Mamba Backbone**: Processes temporal sequences using selective state space models with linear-time complexity
4. **Classification Head**: Outputs class predictions with layer normalization

### Paper Compliance

This implementation strictly follows the paper specifications:

- **Recurrent Connections** (Page 6): Added to spiking front-end for temporal feature extraction
- **LIF Time Constant** (Figure 5): Optimized to τ ≈ 30ms (beta=0.97) for best performance
- **Spike-to-Activation** (Page 7): Implements fixed time window accumulation with firing rate normalization
- **Sequential MNIST** (Table 1): Full support with rate coding conversion from standard MNIST

## Setup

### 1. Environment Setup

The project uses a conda environment named `mambaspike`:

```bash
conda create -n mambaspike python=3.9
conda activate mambaspike
pip install -r requirements.txt
```

### 2. Dataset Preparation

The project supports five neuromorphic datasets:
- **N-MNIST**: Neuromorphic version of MNIST (34x34 resolution)
- **DVS Gesture**: Dynamic hand gestures (128x128 resolution)
- **CIFAR10-DVS**: Neuromorphic version of CIFAR-10 (128x128 resolution)
- **Sequential MNIST**: Standard MNIST converted to spike trains (28x28 resolution)
- **N-TIDIGITS**: Neuromorphic audio dataset (64 frequency channels)

Datasets will be automatically downloaded when running the training scripts.

## Training

Each dataset has a dedicated training script with optimized hyperparameters.

### N-MNIST (Target: 99.5%)

```bash
python train_nmnist.py --batch-size 32 --lr 0.001 --epochs 200
```

### DVS Gesture (Target: 96.8%)

```bash
python train_dvsgesture.py --batch-size 16 --lr 0.001 --epochs 200
```

### CIFAR10-DVS (Target: 78.9%)

```bash
python train_cifar10dvs.py --batch-size 8 --lr 0.001 --epochs 200
```

Note: Uses conservative batch size for memory stability. Dataset size is ~4GB and will be downloaded on first run.

### Sequential MNIST (Target: 99.4%)

```bash
python train_sequential_mnist.py --batch-size 32 --lr 0.001 --epochs 200
```

### N-TIDIGITS (Target: 99.2%)

```bash
python train_ntidigits.py --batch-size 32 --lr 0.001 --epochs 200
```

Note: Audio dataset with 64 frequency channels from cochlea simulation.

### Training Features

- **TensorBoard Logging**: Track training progress in real-time
- **Checkpoint Saving**: Best and latest models saved automatically
- **Device Auto-Detection**: Automatically uses CUDA, MPS, or CPU
- **Progress Tracking**: Detailed console output with training/test metrics
- **JSON Results**: Training configuration and results saved for analysis

### Training Parameters

All training scripts support:
- `--batch-size`: Batch size for training (default varies by dataset)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Maximum training epochs (default: 200)

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint results/nmnist_*/checkpoint_best.pth
```

With additional analysis:
```bash
python evaluate.py --checkpoint results/nmnist_*/checkpoint_best.pth --analyze-temporal --compare-paper
```

Training results are saved in `results/<dataset>_<timestamp>/` with:
- `checkpoint_best.pth`: Best model checkpoint
- `checkpoint_latest.pth`: Latest model checkpoint
- `config.json`: Training configuration
- `final_results.json`: Final accuracy and training time
- `tensorboard/`: TensorBoard logs

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
│   └── dataset_loader.py           # Dataset loading utilities
├── models/
│   └── mamba_spike.py              # Model architecture (paper-compliant)
├── train_nmnist.py                 # N-MNIST training (target: 99.5%)
├── train_dvsgesture.py             # DVS Gesture training (target: 96.8%)
├── train_cifar10dvs.py             # CIFAR10-DVS training (target: 78.9%)
├── train_sequential_mnist.py       # Sequential MNIST training (target: 99.4%)
├── train_ntidigits.py              # N-TIDIGITS training (target: 99.2%)
├── evaluate.py                     # Evaluation script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Key Features

1. **Paper-Compliant Implementation**: Strictly follows all architectural specifications from the CGI 2024 paper
2. **Recurrent Spiking Front-End**: Temporal feature extraction through recurrent connections (30ms time constant)
3. **Efficient Temporal Processing**: Leverages Mamba's selective state spaces for O(L) complexity
4. **Neuromorphic Data Support**: Native processing of event-based data (DVS cameras, Sequential MNIST)
5. **Multi-Scale Architecture**: Supports different input resolutions (28×28 to 128×128)
6. **Flexible Training**: Easy to adapt for different neuromorphic datasets

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