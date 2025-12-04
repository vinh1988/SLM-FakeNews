from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .cleaner import DataCleaner

class DataProcessor:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 32,
        eval_batch_size: int = None,
        random_seed: int = 42,
        clean_data: bool = True,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.random_seed = random_seed
        self.clean_data = clean_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize data cleaner
        if self.clean_data:
            self.cleaner = DataCleaner(dataset_name)
            print(f"✅ Data cleaning enabled for {dataset_name}")
        else:
            self.cleaner = None
            print(f"⚠️  Data cleaning disabled")
        
        # Print batch size configuration
        print(f"📦 Batch sizes - Train: {self.batch_size}, Eval: {self.eval_batch_size}")
        
    def load_dataset(self) -> DatasetDict:
        """Load the specified dataset from Hugging Face."""
        try:
            if self.dataset_name == "welfake":
                # Load dataset with specified cache directory and force redownload
                dataset = load_dataset(
                    "davanstrien/WELFake",
                    cache_dir="/home/vinh/hf_cache",
                    download_mode="force_redownload",
                )
                
                # If it's a Dataset, convert to pandas for splitting
                if isinstance(dataset, Dataset):
                    df = dataset.to_pandas()
                    
                    # Split into train, validation, and test sets
                    train_val, test = train_test_split(
                        df, test_size=0.2, random_state=self.random_seed
                    )
                    train, val = train_test_split(
                        train_val, test_size=0.2, random_state=self.random_seed
                    )
                    
                    # Convert back to Dataset objects
                    return DatasetDict({
                        'train': Dataset.from_pandas(train),
                        'validation': Dataset.from_pandas(val),
                        'test': Dataset.from_pandas(test)
                    })
                    
                return dataset
                    
            elif self.dataset_name == "liar":
                dataset = load_dataset("chengxuphd/liar2")
            elif self.dataset_name == "fakenewsnet":
                dataset = load_dataset("rickstello/FakeNewsNet")
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            return dataset
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {self.dataset_name}: {str(e)}")
    
    def preprocess_function(self, examples):
        """Tokenize the examples with optional data cleaning."""
        # Debug: Print input example keys and types
        if not hasattr(self, '_preprocess_debug_printed'):
            print("\n=== Inside preprocess_function ===")
            print("Input example keys:", list(examples.keys()))
            print("First few items in each key:")
            for k, v in examples.items():
                if k in ['text', 'title', 'content']:
                    print(f"  {k}: {v[:1] if isinstance(v, (list, tuple)) else v}")
            self._preprocess_debug_printed = True
        
        # Apply data cleaning if enabled
        if self.clean_data and self.cleaner is not None:
            cleaned_examples = self.cleaner.clean_batch(examples)
            # Use cleaned content
            text = cleaned_examples.get("content", [])
        else:
            # Get the text column, handle 'text', 'title', 'statement' columns
            text = examples.get("text", examples.get("title", examples.get("statement", examples.get("content", ""))))
        
        # Get labels if they exist (handle different label column names)
        labels = None
        if 'label' in examples:
            labels = examples['label']
        elif 'labels' in examples:
            labels = examples['labels']
        elif 'real' in examples:
            # FakeNewsNet uses 'real' column (1=Real, 0=Fake)
            labels = examples['real']
        
        # Convert LIAR 6-class labels to binary
        # LIAR labels: 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
        # Binary: 0,1,2 -> 0 (Fake), 3,4,5 -> 1 (Real)
        if self.dataset_name == "liar" and labels is not None:
            labels = [0 if l < 3 else 1 for l in labels]
            if not hasattr(self, '_liar_binary_debug_printed'):
                print(f"\n🔄 Converting LIAR 6-class labels to binary (0,1,2→Fake, 3,4,5→Real)")
                self._liar_binary_debug_printed = True
        
        # Ensure text is a list of strings
        if isinstance(text, (list, tuple)):
            text = [str(t) if t is not None else "" for t in text]
        else:
            text = [str(text) if text is not None else ""]
        
        # Debug: Show first cleaned text
        if not hasattr(self, '_cleaned_text_shown') and self.clean_data:
            print(f"\n🧹 First cleaned text sample: {text[0][:200]}...")
            self._cleaned_text_shown = True
        
        # Tokenize the text (don't use return_tensors when batched=True)
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        # Add labels to the output if they exist
        if labels is not None:
            tokenized['labels'] = labels
            
        # Debug: Print output keys and shapes
        if not hasattr(self, '_preprocess_output_debug_printed'):
            print("\nPreprocess function output:")
            for k, v in tokenized.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {v[:2] if isinstance(v, list) else v}")
            
            # Extra debug for LIAR labels
            if self.dataset_name == "liar" and labels is not None:
                print(f"  ⚠️ LIAR labels after conversion: {labels[:10] if isinstance(labels, list) else labels}")
                print(f"  ⚠️ Unique labels: {set(labels) if isinstance(labels, list) else 'N/A'}")
            
            self._preprocess_output_debug_printed = True
            
        return tokenized
    
    def prepare_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test dataloaders."""
        print("\n=== Preparing datasets ===")
        
        # Load dataset
        dataset = self.load_dataset()
        print("\nOriginal dataset structure:", dataset)
        print("Available columns in train split:", dataset['train'].column_names)
        
        # Check if labels exist in the dataset
        label_cols = ['label', 'labels', 'real']
        if not any(col in dataset['train'].column_names for col in label_cols):
            print("\nWARNING: No 'label', 'labels', or 'real' column found in the dataset!")
            print("Dataset columns:", dataset['train'].column_names)
        
        # Ensure we have the required splits
        if not all(split in dataset for split in ['train', 'validation', 'test']):
            print("\nSplitting dataset into train/val/test (75/12.5/12.5)...")
            # First split: 75% train, 25% temp (for val + test)
            train_temp = dataset['train'].train_test_split(
                test_size=0.25,
                seed=self.random_seed
            )
            # Second split: 50/50 of the 25% to get 12.5% val and 12.5% test
            val_test = train_temp['test'].train_test_split(
                test_size=0.5,
                seed=self.random_seed
            )
            dataset = DatasetDict({
                'train': train_temp['train'],
                'validation': val_test['train'],  # 12.5%
                'test': val_test['test']         # 12.5%
            })
        
        print("\nDataset splits sizes:")
        for split_name, split_data in dataset.items():
            print(f"{split_name}: {len(split_data)} examples")
        
        # Print a few examples with labels for verification (BEFORE cleaning)
        print("\n📝 Sample data BEFORE cleaning:")
        for i in range(min(2, len(dataset['train']))):
            example = dataset['train'][i]
            print(f"Example {i}:")
            for col in ['text', 'title', 'statement', 'label', 'labels', 'real']:
                if col in example:
                    val = str(example[col])[:200] + "..." if len(str(example[col])) > 200 else example[col]
                    print(f"  {col}: {val}")
        
        # Preprocess the dataset
        print("\n🔄 Preprocessing dataset...")
        # Remove all columns except the ones needed for training
        columns_to_remove = [col for col in dataset['train'].column_names if col not in ['labels']]
        print(f"📋 Columns to remove during preprocessing: {columns_to_remove}")
        
        tokenized_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=columns_to_remove,
            load_from_cache_file=False,  # Disable cache to ensure fresh preprocessing
            desc="Tokenizing and converting labels"
        )
        
        # Show dataset splits sizes AFTER preprocessing
        print("\n📊 Dataset splits sizes (after preprocessing):")
        for split_name in tokenized_datasets.keys():
            print(f"  {split_name}: {len(tokenized_datasets[split_name])} samples")
        
        # Show cleaned samples AFTER cleaning (before removing content column)
        if self.clean_data:
            print("\n🧹 Sample data AFTER cleaning:")
            for i in range(min(2, len(tokenized_datasets['train']))):
                example = tokenized_datasets['train'][i]
                label = example.get('labels', 'N/A')
                # For LIAR, decode the first few tokens to show cleaned text
                input_ids = example.get('input_ids', [])
                if len(input_ids) > 0:
                    decoded = self.tokenizer.decode(input_ids[:50], skip_special_tokens=True)
                    print(f"  Example {i} [label={label}]: {decoded}...")
                    
        # Debug: Check label distribution in train set
        if self.dataset_name == "liar":
            train_labels = [tokenized_datasets['train'][i]['labels'] for i in range(min(100, len(tokenized_datasets['train'])))]
            unique_labels = set(train_labels)
            print(f"\n⚠️ DEBUG: Unique labels in first 100 train samples: {unique_labels}")
            if any(l >= 2 for l in unique_labels):
                print(f"  ❌ ERROR: Found labels >= 2, conversion failed!")
        
        # Remove 'content' column if it exists (added by cleaner but not needed for training)
        for split in tokenized_datasets.keys():
            if 'content' in tokenized_datasets[split].column_names:
                tokenized_datasets[split] = tokenized_datasets[split].remove_columns(['content'])
        
        # Verify that only needed columns are present after preprocessing
        print("\nAfter preprocessing, dataset features:", tokenized_datasets['train'].features)
        print("Columns in dataset:", tokenized_datasets['train'].column_names)
        
        # Set format for PyTorch
        tokenized_datasets.set_format('torch')
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        print(f"  Train dataloader: batch_size={self.batch_size}")
        print(f"  Val/Test dataloaders: batch_size={self.eval_batch_size}")
        
        train_dataloader = DataLoader(
            tokenized_datasets['train'],
            batch_size=self.batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            tokenized_datasets['validation'],
            batch_size=self.eval_batch_size,
            shuffle=False
        )
        test_dataloader = DataLoader(
            tokenized_datasets['test'],
            batch_size=self.eval_batch_size,
            shuffle=False
        )
        
        # Verify first batch from train dataloader
        print("\nVerifying first training batch...")
        first_batch = next(iter(train_dataloader))
        print("Batch keys:", first_batch.keys())
        for k, v in first_batch.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        
        return train_dataloader, val_dataloader, test_dataloader
