"""
ONNX Export Script

PyTorch DeepFM 모델을 ONNX 형식으로 변환합니다.

Usage:
    python training/src/export/export_onnx.py --model-path training/models/deepfm_emb8_lr0.0005_best.pth
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.models.deepfm_trainer import DeepFMTrainer


def load_processed_data(campaign_id: int,
                       data_base_dir: str = "training/data/processed",
                       neg_pos_ratio: int = None):
    """전처리된 데이터 로드 (vocab size 계산 위해 전체 train 필요)"""
    # Determine directory based on neg_pos_ratio
    if neg_pos_ratio is not None:
        dir_name = f"campaign_{campaign_id}_neg{neg_pos_ratio}"
    else:
        dir_name = f"campaign_{campaign_id}"

    data_path = Path(data_base_dir) / dir_name

    print(f"Loading data from: {data_path}")
    if neg_pos_ratio is not None:
        print(f"Using downsampled data (1:{neg_pos_ratio} ratio)")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at {data_path}. "
            f"Run load_data.py first: python training/src/data/load_data.py "
            f"--campaign {campaign_id}"
            + (f" --neg-pos-ratio {neg_pos_ratio}" if neg_pos_ratio else "")
        )

    with open(data_path / "feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    # Load full train data for vocab size calculation
    train_df = pd.read_csv(data_path / "train.csv")
    # Also load val to ensure correct vocab sizes (important for downsampled data)
    val_df = pd.read_csv(data_path / "val.csv")

    return train_df, val_df, feature_info


def main():
    parser = argparse.ArgumentParser(description='Export DeepFM to ONNX')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output ONNX file path (default: same dir as model)')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--neg-pos-ratio', type=int, default=None,
                       help='Negative to positive ratio for downsampled data (e.g., 100 for 100:1). '
                            'If not set, uses original distribution.')
    parser.add_argument('--dnn-hidden', type=str, default='256,128,64',
                       help='DNN hidden units (comma-separated)')

    args = parser.parse_args()

    # Auto-detect hyperparameters from model filename
    parsed_params = DeepFMTrainer.parse_model_filename(args.model_path)
    embedding_dim = parsed_params['embedding_dim']

    # Parse DNN hidden units
    dnn_hidden_units = tuple(map(int, args.dnn_hidden.split(',')))

    print("=" * 60)
    print("DeepFM ONNX Export")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Embedding dim: {embedding_dim} (auto-detected)")
    print(f"DNN hidden: {dnn_hidden_units}")
    print("=" * 60)

    # Load data (for feature info and shape)
    print("\n=== Loading Feature Info ===")
    train_df, val_df, feature_info = load_processed_data(
        campaign_id=args.campaign,
        data_base_dir=args.data_base_dir,
        neg_pos_ratio=args.neg_pos_ratio
    )

    # Initialize trainer
    trainer = DeepFMTrainer(
        feature_info=feature_info,
        embedding_dim=embedding_dim,
        dnn_hidden_units=dnn_hidden_units,
        device='cpu'  # ONNX export requires CPU
    )

    # Build model and load weights (pass val_df for correct vocab sizes)
    print("\n=== Building Model ===")
    trainer.build_model(train_df, val_df)
    trainer.load_best_model(args.model_path)
    trainer.model.eval()

    # Create dummy input for export
    num_features = len(feature_info['sparse_features']) + len(feature_info['dense_features'])
    dummy_input = torch.randn(1, num_features, dtype=torch.float32)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Total features: {num_features}")

    # Determine output path
    if args.output_path is None:
        model_path = Path(args.model_path)
        args.output_path = str(model_path.parent / model_path.name.replace('.pth', '.onnx'))

    # Export to ONNX
    print(f"\n=== Exporting to ONNX ===")
    print(f"Output: {args.output_path}")

    torch.onnx.export(
        trainer.model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=17,  # Updated to 17 for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("\n✅ ONNX export successful!")
    print(f"ONNX model saved to: {args.output_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed")

    # Print model info
    file_size_mb = Path(args.output_path).stat().st_size / (1024 * 1024)
    print(f"\nModel size: {file_size_mb:.2f} MB")


if __name__ == '__main__':
    main()
