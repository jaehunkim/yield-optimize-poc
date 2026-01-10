"""
DeepFM 학습 스크립트

Usage:
    python training/scripts/train.py --campaign 2259 --epochs 20
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.models.deepfm_trainer import DeepFMTrainer


def load_processed_data(campaign_id: int, num_days: int,
                       data_base_dir: str = "training/data/processed"):
    """전처리된 데이터 로드 (캠페인/days 서브디렉토리)"""
    data_path = Path(data_base_dir) / f"campaign_{campaign_id}" / f"{num_days}days"

    print(f"\n=== Loading Data from {data_path} ===")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at {data_path}. "
            f"Run pipeline.py first: python training/src/validate/pipeline.py "
            f"--campaign {campaign_id} --days {num_days}"
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
    parser = argparse.ArgumentParser(description='Train DeepFM model')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--days', type=int, default=3,
                       help='Number of days')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--save-dir', type=str, default='training/models',
                       help='Model save directory')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=16,
                       help='Embedding dimension')
    parser.add_argument('--dnn-hidden', type=str, default='256,128,64',
                       help='DNN hidden units (comma-separated)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')

    # System optimization
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers (multiprocessing)')

    args = parser.parse_args()

    # Parse DNN hidden units
    dnn_hidden_units = tuple(map(int, args.dnn_hidden.split(',')))

    print("=" * 60)
    print(f"DeepFM Training - Campaign {args.campaign}")
    print("=" * 60)
    print(f"Hyperparameters:")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  DNN hidden: {dnn_hidden_units}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"\nSystem:")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.num_workers}")
    print("=" * 60)

    # Load data
    train_df, val_df, test_df, feature_info = load_processed_data(
        campaign_id=args.campaign,
        num_days=args.days,
        data_base_dir=args.data_base_dir
    )

    # Initialize trainer
    trainer = DeepFMTrainer(
        feature_info=feature_info,
        embedding_dim=args.embedding_dim,
        dnn_hidden_units=dnn_hidden_units,
        learning_rate=args.lr,
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
        save_dir=args.save_dir
    )

    # Save training history
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    history_filename = f'training_history_emb{args.embedding_dim}_lr{args.lr}.json'
    with open(save_path / history_filename, 'w') as f:
        json.dump(history, f, indent=2)

    model_filename = f'deepfm_emb{args.embedding_dim}_lr{args.lr}_best.pth'
    print(f"\nTraining history saved to {save_path / history_filename}")
    print(f"Best model saved to {save_path / model_filename}")


if __name__ == '__main__':
    main()
