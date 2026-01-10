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
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.models.deepfm_trainer import DeepFMTrainer, CTRDataset


def find_latest_model(model_dir: str = "training/models", embedding_dim: int = 8) -> Optional[str]:
    """
    Find the most recent model file for given embedding dimension

    Args:
        model_dir: Directory containing model files
        embedding_dim: Embedding dimension to filter by

    Returns:
        Path to the most recent model file, or None if not found
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return None

    # Find all model files matching pattern: deepfm_emb{embedding_dim}_*_best.pth
    # Supports both old format (deepfm_emb8_lr0.0001_best.pth) and new format (deepfm_emb8_lr0.0001_dnn12864_neg200_best.pth)
    pattern = f"deepfm_emb{embedding_dim}_*_best.pth"
    model_files = list(model_path.glob(pattern))

    if not model_files:
        return None

    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return str(model_files[0])


def load_processed_data(campaign_id: int,
                       data_base_dir: str = "training/data/processed",
                       neg_pos_ratio: int = None):
    """전처리된 데이터 로드 (캠페인별 디렉토리)"""
    # Determine directory based on neg_pos_ratio
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

    return train_df, val_df, test_df, feature_info


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepFM model')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint (if not provided, uses latest model)')
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
    parser.add_argument('--embedding-dim', type=int, default=8,
                       help='Embedding dimension (Phase 1 optimal: 8)')
    parser.add_argument('--dnn-hidden', type=str, default='256,128,64',
                       help='DNN hidden units (comma-separated)')
    parser.add_argument('--neg-pos-ratio', type=int, default=None,
                       help='Negative to positive ratio for downsampled data (e.g., 100 for 100:1). '
                            'If not set, uses original distribution.')

    args = parser.parse_args()

    # Parse DNN hidden units
    dnn_hidden_units = tuple(map(int, args.dnn_hidden.split(',')))

    # Auto-find latest model if not specified
    if args.model_path is None:
        args.model_path = find_latest_model(embedding_dim=args.embedding_dim)
        if args.model_path is None:
            raise FileNotFoundError(
                f"No model found for embedding_dim={args.embedding_dim}. "
                f"Please specify --model-path explicitly."
            )
        print(f"Auto-detected latest model: {args.model_path}")

    print("=" * 60)
    print("DeepFM Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"DNN hidden: {dnn_hidden_units}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load data
    train_df, val_df, test_df, feature_info = load_processed_data(
        campaign_id=args.campaign,
        data_base_dir=args.data_base_dir,
        neg_pos_ratio=args.neg_pos_ratio
    )

    # Initialize trainer
    trainer = DeepFMTrainer(
        feature_info=feature_info,
        embedding_dim=args.embedding_dim,
        dnn_hidden_units=dnn_hidden_units,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers
    )

    # Build model and load weights (pass val_df for correct vocab sizes)
    trainer.build_model(train_df, val_df)
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
