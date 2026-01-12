"""
AutoInt Trainer with GPU acceleration support

AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
- Multi-head Self-Attention for explicit high-order feature interactions
- Larger model compared to DeepFM for optimization effect measurement

Key features:
- GPU acceleration via CUDA (NVIDIA) or MPS (Apple M-series)
- Multiprocessing data loading
- Auto device detection
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
from deepctr_torch.models import AutoInt
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names


class CTRDataset(Dataset):
    """CTR prediction dataset for PyTorch DataLoader"""

    def __init__(self, df: pd.DataFrame, feature_info: Dict):
        self.sparse_features = feature_info['sparse_features']
        self.dense_features = feature_info['dense_features']
        self.target = feature_info['target']

        # Combine all features into single array
        feature_columns = self.sparse_features + self.dense_features
        self.X = df[feature_columns].values.astype(np.float32)
        self.y = df[self.target].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class AutoIntTrainer:
    """AutoInt trainer with GPU optimization"""

    def __init__(self,
                 feature_info: Dict,
                 embedding_dim: int = 64,
                 att_layer_num: int = 3,
                 att_head_num: int = 4,
                 att_res: bool = True,
                 dnn_hidden_units: Tuple[int, ...] = (256, 128),
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
            att_layer_num: Number of attention layers
            att_head_num: Number of attention heads
            att_res: Whether to use residual connection in attention
            dnn_hidden_units: DNN hidden layer sizes
            dnn_dropout: Dropout rate for DNN layers
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            batch_size: Batch size
            device: 'auto', 'mps', 'cuda', or 'cpu'
            num_workers: Number of DataLoader workers
        """
        self.feature_info = feature_info
        self.embedding_dim = embedding_dim
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.att_res = att_res
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_dropout = dnn_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-detect device
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
            vocab_size_key = f'{feat}_vocab_size'
            if vocab_size_key in self.feature_info:
                vocab_size = self.feature_info[vocab_size_key]
            else:
                vocab_size = 10000  # placeholder

            feature_columns.append(
                SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=self.embedding_dim)
            )

        # Dense features
        for feat in self.feature_info['dense_features']:
            feature_columns.append(DenseFeat(feat, 1))

        return feature_columns

    def _update_vocab_sizes(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Update vocabulary sizes from actual data"""
        updated_columns = []
        for col in self.feature_columns:
            if isinstance(col, SparseFeat):
                max_train = train_df[col.name].max()

                if val_df is not None:
                    max_val = val_df[col.name].max()
                    actual_vocab_size = max(max_train, max_val) + 1
                else:
                    actual_vocab_size = max_train + 1

                updated_columns.append(
                    SparseFeat(col.name, vocabulary_size=actual_vocab_size,
                             embedding_dim=col.embedding_dim)
                )
            else:
                updated_columns.append(col)
        self.feature_columns = updated_columns

    def build_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Build AutoInt model"""
        # Update vocab sizes from data
        self._update_vocab_sizes(train_df, val_df)

        # Create AutoInt model
        self.model = AutoInt(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            att_layer_num=self.att_layer_num,
            att_head_num=self.att_head_num,
            att_res=self.att_res,
            dnn_hidden_units=self.dnn_hidden_units,
            dnn_dropout=self.dnn_dropout,
            device=self.device
        )

        # Optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / 1024 / 1024  # FP32 = 4 bytes

        print(f"\nAutoInt Model built:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Estimated model size: {model_size_mb:.2f} MB")
        print(f"  Attention layers: {self.att_layer_num}")
        print(f"  Attention heads: {self.att_head_num}")
        print(f"  Attention residual: {self.att_res}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in train_loader:
            x = batch_x.to(self.device)
            y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x).squeeze()

            loss = nn.functional.binary_cross_entropy(y_pred, y)

            loss.backward()
            self.optimizer.step()

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
                x = batch_x.to(self.device)
                y = batch_y.to(self.device)

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
            save_dir: str = 'training/models',
            neg_pos_ratio: int = None):
        """Train AutoInt model"""

        if self.model is None:
            self.build_model(train_df, val_df)

        # Create datasets
        train_dataset = CTRDataset(train_df, self.feature_info)
        val_dataset = CTRDataset(val_df, self.feature_info)

        # Create dataloaders
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

            train_loss, train_auc = self.train_epoch(train_loader)
            val_auc, val_logloss, val_ctr = self.evaluate(val_loader)

            epoch_time = time.time() - start_time

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

                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)

                # Build filename with hyperparameters
                dnn_str = "".join(str(h) for h in self.dnn_hidden_units)
                if neg_pos_ratio is not None:
                    model_filename = f'autoint_emb{self.embedding_dim}_att{self.att_layer_num}x{self.att_head_num}_dnn{dnn_str}_neg{neg_pos_ratio}_best.pth'
                else:
                    model_filename = f'autoint_emb{self.embedding_dim}_att{self.att_layer_num}x{self.att_head_num}_dnn{dnn_str}_best.pth'

                torch.save(self.model.state_dict(), save_path / model_filename)
                print(f"  -> New best model saved: {model_filename} (AUC: {best_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nTraining complete! Best Val AUC: {best_auc:.4f}")

        return history

    def load_best_model(self, model_path: str):
        """Load best model from checkpoint"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    @staticmethod
    def parse_model_filename(model_path: str):
        """
        Parse hyperparameters from model filename

        Expected format: autoint_emb{dim}_att{layers}x{heads}_dnn{units}_neg{ratio}_best.pth
        """
        import re
        filename = Path(model_path).name

        emb_match = re.search(r'emb(\d+)', filename)
        embedding_dim = int(emb_match.group(1)) if emb_match else 64

        att_match = re.search(r'att(\d+)x(\d+)', filename)
        att_layer_num = int(att_match.group(1)) if att_match else 3
        att_head_num = int(att_match.group(2)) if att_match else 4

        return {
            'embedding_dim': embedding_dim,
            'att_layer_num': att_layer_num,
            'att_head_num': att_head_num
        }
