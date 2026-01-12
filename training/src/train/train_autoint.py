"""
AutoInt 학습 스크립트

Usage:
    python training/src/train/train_autoint.py --campaign 2259 --epochs 20

AutoInt uses Multi-head Self-Attention for feature interaction learning.
Larger model compared to DeepFM for optimization effect measurement.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from training.src.models.autoint_trainer import AutoIntTrainer


def load_processed_data(campaign_id: int,
                       data_base_dir: str = "training/data/processed",
                       neg_pos_ratio: int = None):
    """전처리된 데이터 로드 (캠페인별 디렉토리)"""
    if neg_pos_ratio is not None:
        dir_name = f"campaign_{campaign_id}_neg{neg_pos_ratio}"
    else:
        dir_name = f"campaign_{campaign_id}"

    data_path = Path(data_base_dir) / dir_name

    print(f"\n=== Loading Data from {data_path} ===")
    if neg_pos_ratio is not None:
        print(f"Using downsampled data (1:{neg_pos_ratio} ratio)")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at {data_path}. "
            f"Run load_data.py first: python training/src/data/load_data.py "
            f"--campaign {campaign_id}"
            + (f" --neg-pos-ratio {neg_pos_ratio}" if neg_pos_ratio else "")
        )

    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    with open(data_path / "feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_info['sparse_features'])} sparse + "
          f"{len(feature_info['dense_features'])} dense")
    print(f"Click rate - Train: {train_df['click'].mean():.4f}, "
          f"Val: {val_df['click'].mean():.4f}, "
          f"Test: {test_df['click'].mean():.4f}")

    return train_df, val_df, test_df, feature_info


def main():
    parser = argparse.ArgumentParser(description='Train AutoInt model')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--save-dir', type=str, default='training/models',
                       help='Model save directory')
    parser.add_argument('--neg-pos-ratio', type=int, default=None,
                       help='Negative to positive ratio for downsampled data')

    # AutoInt-specific hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension (larger than DeepFM for bigger model)')
    parser.add_argument('--att-layer-num', type=int, default=3,
                       help='Number of attention layers')
    parser.add_argument('--att-head-num', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--att-res', action='store_true', default=True,
                       help='Use residual connection in attention')
    parser.add_argument('--dnn-hidden', type=str, default='256,128',
                       help='DNN hidden units (comma-separated)')
    parser.add_argument('--dnn-dropout', type=float, default=0.5,
                       help='Dropout rate for DNN layers')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='L2 regularization strength')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')

    # System optimization
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers')

    args = parser.parse_args()

    # Parse DNN hidden units
    dnn_hidden_units = tuple(map(int, args.dnn_hidden.split(',')))

    print("=" * 70)
    print(f"AutoInt Training - Campaign {args.campaign}")
    print("=" * 70)
    print(f"Model Architecture:")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Attention layers: {args.att_layer_num}")
    print(f"  Attention heads: {args.att_head_num}")
    print(f"  Attention residual: {args.att_res}")
    print(f"  DNN hidden: {dnn_hidden_units}")
    print(f"  DNN dropout: {args.dnn_dropout}")
    print(f"\nTraining:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"\nSystem:")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.num_workers}")
    print("=" * 70)

    # Load data
    train_df, val_df, test_df, feature_info = load_processed_data(
        campaign_id=args.campaign,
        data_base_dir=args.data_base_dir,
        neg_pos_ratio=args.neg_pos_ratio
    )

    # Initialize trainer
    trainer = AutoIntTrainer(
        feature_info=feature_info,
        embedding_dim=args.embedding_dim,
        att_layer_num=args.att_layer_num,
        att_head_num=args.att_head_num,
        att_res=args.att_res,
        dnn_hidden_units=dnn_hidden_units,
        dnn_dropout=args.dnn_dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers
    )

    # Train
    history = trainer.fit(
        train_df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_dir=args.save_dir,
        neg_pos_ratio=args.neg_pos_ratio
    )

    # Save training history
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Build filenames
    dnn_str = "".join(str(h) for h in dnn_hidden_units)
    if args.neg_pos_ratio is not None:
        history_filename = f'autoint_history_emb{args.embedding_dim}_att{args.att_layer_num}x{args.att_head_num}_dnn{dnn_str}_neg{args.neg_pos_ratio}.json'
        model_filename = f'autoint_emb{args.embedding_dim}_att{args.att_layer_num}x{args.att_head_num}_dnn{dnn_str}_neg{args.neg_pos_ratio}_best.pth'
    else:
        history_filename = f'autoint_history_emb{args.embedding_dim}_att{args.att_layer_num}x{args.att_head_num}_dnn{dnn_str}.json'
        model_filename = f'autoint_emb{args.embedding_dim}_att{args.att_layer_num}x{args.att_head_num}_dnn{dnn_str}_best.pth'

    with open(save_path / history_filename, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining history saved to {save_path / history_filename}")
    print(f"Best model saved to {save_path / model_filename}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on Test Set...")
    print("=" * 70)

    from training.src.models.autoint_trainer import CTRDataset
    from torch.utils.data import DataLoader

    # Load best model
    trainer.load_best_model(save_path / model_filename)

    # Create test loader
    test_dataset = CTRDataset(test_df, feature_info)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_auc, test_logloss, test_ctr = trainer.evaluate(test_loader)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test LogLoss: {test_logloss:.4f}")
    print(f"Test CTR: {test_ctr:.4f}")

    # Save final results
    results = {
        'model': 'AutoInt',
        'campaign': args.campaign,
        'hyperparameters': {
            'embedding_dim': args.embedding_dim,
            'att_layer_num': args.att_layer_num,
            'att_head_num': args.att_head_num,
            'att_res': args.att_res,
            'dnn_hidden_units': dnn_hidden_units,
            'dnn_dropout': args.dnn_dropout,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'neg_pos_ratio': args.neg_pos_ratio
        },
        'test_metrics': {
            'auc': float(test_auc),
            'logloss': float(test_logloss),
            'ctr': float(test_ctr)
        },
        'training_history': history
    }

    results_filename = f'autoint_results_emb{args.embedding_dim}_att{args.att_layer_num}x{args.att_head_num}.json'
    with open(save_path / results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_path / results_filename}")


if __name__ == '__main__':
    main()
