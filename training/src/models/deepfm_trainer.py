"""
DeepFM Trainer with GPU acceleration support

Key optimizations:
- GPU acceleration via CUDA (NVIDIA) or MPS (Apple M-series)
- Multiprocessing data loading (num_workers > 0)
- Pin memory for faster CUDA data transfer
- Auto device detection (CUDA > MPS > CPU)
"""

import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import Dataset, DataLoader
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names


class CTRDataset(Dataset):
    """CTR prediction dataset for PyTorch DataLoader"""

    def __init__(self, df: pd.DataFrame, feature_info: Dict):
        self.sparse_features = feature_info['sparse_features']
        self.dense_features = feature_info['dense_features']
        self.target = feature_info['target']

        # Combine all features into single array (DeepCTR expects concatenated input)
        feature_columns = self.sparse_features + self.dense_features
        self.X = df[feature_columns].values.astype(np.float32)
        self.y = df[self.target].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class DeepFMTrainer:
    """DeepFM trainer with M2 optimization"""

    def __init__(self,
                 feature_info: Dict,
                 embedding_dim: int = 16,
                 dnn_hidden_units: Tuple[int, ...] = (256, 128, 64),
                 dnn_dropout: float = 0.5,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 512,
                 device: str = 'auto',
                 num_workers: int = 4):
        """
        Args:
            feature_info: Feature metadata from load_data.py
            embedding_dim: Embedding dimension for sparse features
            dnn_hidden_units: DNN hidden layer sizes
            dnn_dropout: Dropout rate for DNN layers
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            batch_size: Batch size
            device: 'auto', 'mps', 'cuda', or 'cpu'
            num_workers: Number of DataLoader workers (multiprocessing)
        """
        self.feature_info = feature_info
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_dropout = dnn_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-detect device (prioritize CUDA over MPS)
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Using CUDA - {gpu_name} ({gpu_memory:.1f} GB)")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("Using MPS (Apple M-series GPU acceleration)")
            else:
                self.device = torch.device('cpu')
                print("Using CPU")
        else:
            self.device = torch.device(device)
            if device == 'cuda' and torch.cuda.is_available():
                print(f"Using CUDA - {torch.cuda.get_device_name(0)}")

        # Build feature columns for DeepCTR
        self.feature_columns = self._build_feature_columns()

        # Initialize model
        self.model = None
        self.optimizer = None

    def _build_feature_columns(self):
        """Build feature columns for DeepCTR"""
        feature_columns = []

        # Sparse features
        for feat in self.feature_info['sparse_features']:
            # Get vocabulary size from feature_info
            vocab_size_key = f'{feat}_vocab_size'
            if vocab_size_key in self.feature_info:
                vocab_size = self.feature_info[vocab_size_key]
            else:
                # Fallback: assume max value + 1 (will be set during data loading)
                vocab_size = 10000  # placeholder

            feature_columns.append(
                SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=self.embedding_dim)
            )

        # Dense features
        for feat in self.feature_info['dense_features']:
            feature_columns.append(DenseFeat(feat, 1))

        return feature_columns

    def _update_vocab_sizes(self, train_df: pd.DataFrame):
        """Update vocabulary sizes from actual data"""
        updated_columns = []
        for col in self.feature_columns:
            if isinstance(col, SparseFeat):
                actual_vocab_size = train_df[col.name].max() + 1
                updated_columns.append(
                    SparseFeat(col.name, vocabulary_size=actual_vocab_size,
                             embedding_dim=col.embedding_dim)
                )
            else:
                updated_columns.append(col)
        self.feature_columns = updated_columns

    def build_model(self, train_df: pd.DataFrame):
        """Build DeepFM model"""
        # Update vocab sizes from data
        self._update_vocab_sizes(train_df)

        # Create model with dropout
        self.model = DeepFM(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            dnn_hidden_units=self.dnn_hidden_units,
            dnn_dropout=self.dnn_dropout,
            device=self.device
        )

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        print(f"\nModel built with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in train_loader:
            # Move to device
            x = batch_x.to(self.device)
            y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x).squeeze()

            # Loss
            loss = nn.functional.binary_cross_entropy(y_pred, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * len(y)
            total_samples += len(y)

            all_preds.extend(y_pred.detach().cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / total_samples
        auc = roc_auc_score(all_labels, all_preds)

        return avg_loss, auc

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move to device
                x = batch_x.to(self.device)
                y = batch_y.to(self.device)

                # Forward pass
                y_pred = self.model(x).squeeze()

                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, all_preds)

        return auc, logloss, np.mean(all_labels)

    def fit(self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            epochs: int = 10,
            early_stopping_patience: int = 3,
            save_dir: str = 'training/models'):
        """Train DeepFM model"""

        # Build model
        if self.model is None:
            self.build_model(train_df)

        # Create datasets
        train_dataset = CTRDataset(train_df, self.feature_info)
        val_dataset = CTRDataset(val_df, self.feature_info)

        # Create dataloaders with multiprocessing
        # pin_memory enables faster CUDA data transfer (not supported by MPS)
        use_pin_memory = self.device.type == 'cuda'

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )

        # Training loop
        best_auc = 0
        patience_counter = 0
        history = []

        print(f"\nTraining for {epochs} epochs...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Batch size: {self.batch_size}, Workers: {self.num_workers}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss, train_auc = self.train_epoch(train_loader)

            # Validate
            val_auc, val_logloss, val_ctr = self.evaluate(val_loader)

            epoch_time = time.time() - start_time

            # Log
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                  f"Val AUC: {val_auc:.4f}, Val LogLoss: {val_logloss:.4f}")

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'val_logloss': val_logloss,
                'time': epoch_time
            })

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save best model with hyperparameters in filename
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                model_filename = f'deepfm_emb{self.embedding_dim}_lr{self.learning_rate}_best.pth'
                torch.save(self.model.state_dict(), save_path / model_filename)
                print(f"  -> New best model saved: {model_filename} (AUC: {best_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nTraining complete! Best Val AUC: {best_auc:.4f}")

        return history

    def load_best_model(self, model_path: str = 'training/models/deepfm_best.pth'):
        """Load best model from checkpoint"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    @staticmethod
    def parse_model_filename(model_path: str):
        """
        Parse hyperparameters from model filename

        Expected format: deepfm_emb{dim}_lr{lr}_best.pth
        Example: deepfm_emb8_lr0.0005_best.pth -> {'embedding_dim': 8, 'lr': 0.0005}
        """
        import re
        filename = Path(model_path).name

        # Parse embedding dimension
        emb_match = re.search(r'emb(\d+)', filename)
        embedding_dim = int(emb_match.group(1)) if emb_match else 16

        # Parse learning rate
        lr_match = re.search(r'lr([\d.]+)', filename)
        lr = float(lr_match.group(1)) if lr_match else 0.001

        return {
            'embedding_dim': embedding_dim,
            'lr': lr
        }
