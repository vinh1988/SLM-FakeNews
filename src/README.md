# Fake News Detection with Adapter-based Transformers

This project implements adapter-based transformer models for fake news detection across multiple datasets.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Available Datasets

- `welfake`: WELFake dataset (54.5% Fake, 45.5% Real)
- `liar`: LIAR dataset (39.4% Fake, 60.6% Real)
- `fakenewsnet`: FakeNewsNet dataset (24.2% Fake, 75.8% Real)

## Available Models

- `distilbert`: DistilBERT base uncased
- `minilm`: Microsoft MiniLM-L12-H384-uncased
- `albert`: ALBERT base v2

## Training

### Basic Usage

Train a model on a specific dataset:
```bash
python main.py --dataset DATASET_NAME --model MODEL_NAME --output_dir ./results
```

### Examples

1. Train DistilBERT on WELFake dataset:
   ```bash
   python main.py --dataset welfake --model distilbert --output_dir ./results
   ```

2. Train with custom hyperparameters:
   ```bash
   python main.py \
     --dataset liar \
     --model minilm \
     --batch_size 16 \
     --num_epochs 5 \
     --learning_rate 2e-5 \
     --output_dir ./results
   ```

3. Train on all datasets and models (be careful with resources):
   ```bash
   python main.py --output_dir ./results
   ```

### Arguments

- `--dataset`: Dataset name (welfake, liar, fakenewsnet). If None, runs all datasets.
- `--model`: Model type (distilbert, minilm, albert). If None, runs all models.
- `--output_dir`: Directory to save model checkpoints and results (default: "./results").
- `--batch_size`: Batch size for training and evaluation (default: 32).
- `--num_epochs`: Number of training epochs (default: 3).
- `--learning_rate`: Learning rate for the optimizer (default: 5e-5).
- `--max_length`: Maximum sequence length (default: 384).
- `--seed`: Random seed (default: 42).

## Output

The training script will save the following in the specified output directory:
- Model checkpoints
- Training logs
- Evaluation metrics
- Predictions on test set

## Notes

- The script automatically handles class imbalance using class weights.
- Training progress is logged to the console and saved to a file.
- The best model checkpoint is saved based on validation performance.
