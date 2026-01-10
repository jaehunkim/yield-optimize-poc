"""
DeepFM 평가 스크립트

Usage:
    python training/scripts/evaluate.py --model-path training/models/deepfm_best.pth
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.models.deepfm_trainer import DeepFMTrainer, CTRDataset


def load_processed_data(campaign_id: int, num_days: int,
                       data_base_dir: str = "training/data/processed"):
    """전처리된 데이터 로드 (캠페인/days 서브디렉토리)"""
    data_path = Path(data_base_dir) / f"campaign_{campaign_id}" / f"{num_days}days"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    with open(data_path / "feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    return train_df, val_df, test_df, feature_info


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepFM model')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--days', type=int, default=3,
                       help='Number of days')
    parser.add_argument('--model-path', type=str, default='training/models/deepfm_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--output', type=str, default='training/results/evaluation.json',
                       help='Output file for results')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--embedding-dim', type=int, default=None,
                       help='Embedding dimension (if None, auto-detect from model filename)')

    args = parser.parse_args()

    # Auto-detect hyperparameters from model filename if not specified
    if args.embedding_dim is None:
        parsed_params = DeepFMTrainer.parse_model_filename(args.model_path)
        args.embedding_dim = parsed_params['embedding_dim']
        print(f"Auto-detected from filename: embedding_dim={args.embedding_dim}")

    print("=" * 60)
    print("DeepFM Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load data
    print("\n=== Loading Data ===")
    train_df, val_df, test_df, feature_info = load_processed_data(
        campaign_id=args.campaign,
        num_days=args.days,
        data_base_dir=args.data_base_dir
    )

    # Initialize trainer
    trainer = DeepFMTrainer(
        feature_info=feature_info,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers
    )

    # Build model and load weights
    trainer.build_model(train_df)
    trainer.load_best_model(args.model_path)

    # Evaluate on all sets
    results = {}

    # pin_memory only for CUDA (MPS doesn't support it yet)
    use_pin_memory = trainer.device.type == 'cuda'

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n=== Evaluating on {name.upper()} set ===")

        dataset = CTRDataset(df, feature_info)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory
        )

        auc, logloss, ctr = trainer.evaluate(loader)

        print(f"{name.capitalize()} - AUC: {auc:.4f}, LogLoss: {logloss:.4f}, CTR: {ctr:.4f}")

        results[name] = {
            'auc': float(auc),
            'logloss': float(logloss),
            'ctr': float(ctr),
            'samples': len(df)
        }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results saved to {output_path} ===")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test AUC: {results['test']['auc']:.4f}")
    print(f"Test LogLoss: {results['test']['logloss']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
