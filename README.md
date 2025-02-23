# Self-Supervised Transformers for Long-Term Prediction of Landsat NDVI Time Series

Official implementation of the paper "Self-Supervised Transformers for Long-Term Prediction of Landsat NDVI Time Series" (ICPRAM 2025).

## Abstract

This repository contains the implementation of a novel self-supervised transformer architecture for predicting long-term Normalized Difference Vegetation Index (NDVI) time series from Landsat satellite data. Our approach leverages temporal positional encoding and self-attention mechanisms to capture complex seasonal patterns and long-term dependencies in vegetation dynamics.

## Model Architecture

The model consists of:
- Temporal Positional NDVI Transformer with continuous temporal encoding
- Self-attention mechanism for capturing long-range dependencies
- Sequence-aware positional encoding
- Multi-head attention for parallel processing of temporal features

## Dataset

The model is trained on Landsat NDVI time series data from 1984 to 2024, consisting of bi-seasonal (summer/winter) NDVI measurements. The dataset should be structured as a multi-band GeoTIFF file where each band represents an NDVI measurement at a specific timestamp.

### Data Format
- Input: Multi-band GeoTIFF file
- Each band: NDVI measurement (-1 to 1 range)
- Temporal resolution: Bi-seasonal (summer/winter)
- Spatial resolution: 30m (Landsat resolution)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SST-LTP.git
cd SST-LTP
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py --dataset_path path/to/your/landsat_data.tif \
                --window_size 40 \
                --future_window_size 10 \
                --batch_size 32 \
                --epochs 100 \
                --name experiment_name
```

### Evaluation

To evaluate a trained model:

```bash
python eval.py --experiment experiment_name
```

### Testing

To test the model's predictions:

```bash
python test.py --experiment experiment_name
```

## Model Parameters

- `window_size`: Number of timesteps to look at (default: 40)
- `future_window_size`: Number of timesteps to predict ahead (default: 10)
- `embedding_dim`: Dimension of the embedding (default: 256)
- `num_encoder_layers`: Number of transformer encoder layers (default: 3)
- `attn_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)

## Visualization

The repository includes a Jupyter notebook (`visualize.ipynb`) for visualizing:
- Training and validation losses
- NDVI predictions vs. ground truth
- Attention patterns
- Temporal embeddings

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{faran2025self,
  title={Self-Supervised Transformers for Long-Term Prediction of Landsat NDVI Time Series},
  author={Faran, Ido and Netanyahu, Nathan S. and Roitberg, Elena and Shoshany, Maxim},
  booktitle={Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods (ICPRAM)},
  year={2025}
}
```

## Contact

For questions or feedback about this project, please contact:

Ido Faran - [faranidof@gmail.com](mailto:faranidof@gmail.com)