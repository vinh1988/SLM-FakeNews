import argparse
import os
import json
import torch
from typing import Dict, Any, List
from pathlib import Path

from config.datasets import get_dataset_config, get_model_name, get_class_weights
from models.adapter_bert import AdapterBERT
from data import DataProcessor
from training.trainer import AdapterTrainer
from torch.optim import AdamW

def parse_args():
    parser = argparse.ArgumentParser(description="Train adapter-based models for fake news detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (welfake, fake_news, fakenewsnet). If None, runs all datasets."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model type (distilbert, minilm, albert). If None, runs all models."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save model checkpoints and results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def train_and_evaluate(
    dataset_name: str,
    model_type: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    seed: int
) -> Dict[str, Any]:
    """Train and evaluate a model on a dataset."""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Check and configure CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Device Configuration:")
    print(f"{'='*50}")
    print(f"  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Current GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    else:
        print(f"  WARNING: CUDA is not available. Training will be slow on CPU.")
    print(f"{'='*50}\n")
    
    # Get model name
    model_name = get_model_name(model_type)
    
    # Create output directory
    exp_dir = os.path.join(output_dir, f"{dataset_name}_{model_type}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize data processor with different batch sizes for train and eval
    processor = DataProcessor(
        dataset_name=dataset_name,
        model_name=model_name,
        max_length=max_length,
        batch_size=16,  # Training batch size
        eval_batch_size=32,  # Evaluation batch size
        random_seed=seed
    )
    
    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = processor.prepare_datasets()
    
    # Get number of samples
    num_train_samples = len(train_dataloader.dataset)
    num_test_samples = len(test_dataloader.dataset)
    print(f"📊 Train samples: {num_train_samples}, Test samples: {num_test_samples}")
    
    # Initialize model
    model = AdapterBERT(
        model_name=model_name,
        num_labels=2,  # Binary classification
        adapter_config={
            "adapter_size": 64,
            "adapter_initializer_range": 1e-4,
            "adapter_layer_norm_eps": 1e-12,
        }
    )
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Get class weights for imbalanced datasets
    class_weights = get_class_weights(dataset_name)
    
    # Initialize trainer (use test set for evaluation, no checkpoint saving)
    trainer = AdapterTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,  # Use test set for evaluation
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_dir=exp_dir,
        save_checkpoints=False,  # Don't save model checkpoints
        class_weights=class_weights,  # Apply class weights for imbalanced data
    )
    
    # Train the model
    print(f"Training {model_type} on {dataset_name}...")
    best_metrics = trainer.train()
    
    # Format results with requested metrics (evaluated on test set)
    results = {
        "dataset": dataset_name,
        "model": model_type,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
        "metrics": {
            "Accuracy": best_metrics['accuracy'],
            "Precision": best_metrics['precision'],
            "Recall": best_metrics['recall'],
            "F1": best_metrics['f1'],
            "AUC": best_metrics['auc'],
            "samples_per_sec": best_metrics['samples_per_sec'],
            "Train_Time_s": best_metrics['train_time_s'],
            "Test_Loss": best_metrics['eval_loss'],
            "Test_Steps_per_sec": best_metrics['steps_per_sec'],
        },
        "hyperparameters": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "seed": seed,
        }
    }
    
    # Save results to file
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    args = parse_args()
    
    # Define datasets and models to run
    datasets = ["welfake", "liar", "fakenewsnet"]
    models = ["distilbert", "minilm", "albert"]
    
    # Filter if specific dataset/model is provided
    if args.dataset is not None:
        datasets = [args.dataset]
    if args.model is not None:
        models = [args.model]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    all_results = []
    for dataset in datasets:
        for model_type in models:
            try:
                print(f"\n{'='*50}")
                print(f"Dataset: {dataset}, Model: {model_type}")
                print(f"{'='*50}")
                
                result = train_and_evaluate(
                    dataset_name=dataset,
                    model_type=model_type,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    max_length=args.max_length,
                    seed=args.seed
                )
                all_results.append(result)
                
                # Print results summary (evaluated on test set)
                print(f"\n{'='*50}")
                print(f"Final Results for {dataset} - {model_type} (Test Set):")
                print(f"{'='*50}")
                metrics = result['metrics']
                print(f"  Accuracy:           {metrics['Accuracy']:.4f}")
                print(f"  Precision:          {metrics['Precision']:.4f}")
                print(f"  Recall:             {metrics['Recall']:.4f}")
                print(f"  F1:                 {metrics['F1']:.4f}")
                print(f"  AUC:                {metrics['AUC']:.4f}")
                print(f"  samples_per_sec:    {metrics['samples_per_sec']:.2f}")
                print(f"  Train_Time_s:       {metrics['Train_Time_s']:.2f}")
                print(f"  Test_Loss:          {metrics['Test_Loss']:.4f}")
                print(f"  Test_Steps_per_sec: {metrics['Test_Steps_per_sec']:.2f}")
                print(f"{'='*50}")
                
            except Exception as e:
                print(f"Error running {model_type} on {dataset}: {str(e)}")
                continue
    
    # Save all results to a single file
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()
