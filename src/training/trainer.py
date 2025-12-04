import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class AdapterTrainer:
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: str = "./checkpoints",
        log_interval: int = 10,
        save_checkpoints: bool = False,
        class_weights: Tuple[float, float] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_checkpoints = save_checkpoints
        self.global_step = 0
        
        # Setup class weights for imbalanced datasets
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
            print(f"⚖️ Using class weights: {class_weights}")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.total_train_time = 0.0
        self.total_train_samples = 0
        self.epoch_start_time = None
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)

    def train_epoch(self, epoch: int):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        epoch_samples = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
            disable=False,
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            epoch_samples += batch['input_ids'].size(0)
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Use custom loss function with class weights
            logits = outputs.logits
            labels = batch['labels']
            loss = self.loss_fn(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update parameters
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            
            if self.global_step % self.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': avg_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        epoch_time = time.time() - epoch_start_time
        self.total_train_time += epoch_time
        self.total_train_samples += epoch_samples
        
        return total_loss / len(self.train_dataloader)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        eval_start_time = time.time()
        eval_samples = 0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                eval_samples += batch['input_ids'].size(0)
                eval_steps += 1
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Use custom loss function with class weights
                labels = batch['labels']
                loss = self.loss_fn(logits, labels)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                # Update metrics
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        eval_time = time.time() - eval_start_time
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Calculate AUC
        try:
            all_probs_array = np.array(all_probs)
            # For binary classification, use the probability of the positive class
            if all_probs_array.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs_array[:, 1])
            else:
                # For multi-class, use one-vs-rest
                auc = roc_auc_score(all_labels, all_probs_array, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {str(e)}")
            auc = 0.0
        
        return {
            'eval_loss': total_loss / len(self.val_dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'samples_per_sec': eval_samples / eval_time if eval_time > 0 else 0.0,
            'steps_per_sec': eval_steps / eval_time if eval_time > 0 else 0.0,
        }
    
    def train(self) -> Dict[str, float]:
        """Train the model for the specified number of epochs."""
        best_metric = 0.0
        best_eval_metrics = None
        
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on validation set
            eval_metrics = self.evaluate()
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}:")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Test loss: {eval_metrics['eval_loss']:.4f}")
            print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Precision: {eval_metrics['precision']:.4f}")
            print(f"  Recall: {eval_metrics['recall']:.4f}")
            print(f"  F1 Score: {eval_metrics['f1']:.4f}")
            print(f"  AUC: {eval_metrics['auc']:.4f}")
            print(f"  Test samples/sec: {eval_metrics['samples_per_sec']:.2f}")
            print(f"  Test steps/sec: {eval_metrics['steps_per_sec']:.2f}")
            
            # Track best model
            if eval_metrics['f1'] > best_metric:
                best_metric = eval_metrics['f1']
                best_eval_metrics = eval_metrics.copy()
                if self.save_checkpoints:
                    self.save_model(os.path.join(self.save_dir, "best_model"))
                    print(f"  New best model saved with F1: {best_metric:.4f}")
                else:
                    print(f"  New best F1: {best_metric:.4f}")
            
            # Save checkpoint (only if enabled)
            if self.save_checkpoints:
                self.save_model(os.path.join(self.save_dir, f"checkpoint-{epoch}"))
        
        # Calculate training throughput
        train_samples_per_sec = self.total_train_samples / self.total_train_time if self.total_train_time > 0 else 0.0
        
        # Add training metrics to best evaluation metrics
        if best_eval_metrics is not None:
            best_eval_metrics['train_samples_per_sec'] = train_samples_per_sec
            best_eval_metrics['train_time_s'] = self.total_train_time
        
        # Print GPU memory usage if using CUDA
        if self.device.type == 'cuda':
            print(f"\nGPU Memory Usage:")
            print(f"  Max memory allocated: {torch.cuda.max_memory_allocated(self.device) / 1024**2:.2f} MB")
            print(f"  Max memory cached: {torch.cuda.max_memory_reserved(self.device) / 1024**2:.2f} MB")
        
        return best_eval_metrics
    
    def save_model(self, output_dir: str):
        """Save the model and optimizer state."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(output_dir, "optimizer.pt")
        )
    
    def load_model(self, model_path: str):
        """Load a saved model and optimizer state."""
        self.model = self.model.from_pretrained(model_path)
        optimizer_path = os.path.join(model_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
