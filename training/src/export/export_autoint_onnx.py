"""
AutoInt ONNX Export Script

PyTorch AutoInt 모델을 ONNX 형식으로 변환합니다.

Usage:
    python training/src/export/export_autoint_onnx.py \
        --model-path training/models/autoint_emb64_att3x4_dnn256128_neg150_best.pth
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from training.src.models.autoint_trainer import AutoIntTrainer


def load_processed_data(campaign_id: int,
                       data_base_dir: str = "training/data/processed",
                       neg_pos_ratio: int = None):
    """전처리된 데이터 로드"""
    if neg_pos_ratio is not None:
        dir_name = f"campaign_{campaign_id}_neg{neg_pos_ratio}"
    else:
        dir_name = f"campaign_{campaign_id}"

    data_path = Path(data_base_dir) / dir_name

    print(f"Loading data from: {data_path}")
    if neg_pos_ratio is not None:
        print(f"Using downsampled data (1:{neg_pos_ratio} ratio)")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    with open(data_path / "feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")

    return train_df, val_df, feature_info


def main():
    parser = argparse.ArgumentParser(description='Export AutoInt to ONNX')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output ONNX file path (default: same dir as model)')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--neg-pos-ratio', type=int, default=None,
                       help='Negative to positive ratio for downsampled data')

    # AutoInt hyperparameters (auto-detect from filename or specify)
    parser.add_argument('--embedding-dim', type=int, default=None,
                       help='Embedding dimension (auto-detect if not specified)')
    parser.add_argument('--att-layer-num', type=int, default=None,
                       help='Number of attention layers')
    parser.add_argument('--att-head-num', type=int, default=None,
                       help='Number of attention heads')
    parser.add_argument('--dnn-hidden', type=str, default='256,128',
                       help='DNN hidden units (comma-separated)')

    args = parser.parse_args()

    # Auto-detect hyperparameters from model filename
    parsed_params = AutoIntTrainer.parse_model_filename(args.model_path)

    embedding_dim = args.embedding_dim or parsed_params['embedding_dim']
    att_layer_num = args.att_layer_num or parsed_params['att_layer_num']
    att_head_num = args.att_head_num or parsed_params['att_head_num']
    dnn_hidden_units = tuple(map(int, args.dnn_hidden.split(',')))

    print("=" * 60)
    print("AutoInt ONNX Export")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Attention layers: {att_layer_num}")
    print(f"Attention heads: {att_head_num}")
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
    trainer = AutoIntTrainer(
        feature_info=feature_info,
        embedding_dim=embedding_dim,
        att_layer_num=att_layer_num,
        att_head_num=att_head_num,
        dnn_hidden_units=dnn_hidden_units,
        device='cpu'  # ONNX export requires CPU
    )

    # Build model and load weights
    print("\n=== Building Model ===")
    trainer.build_model(train_df, val_df)
    trainer.load_best_model(args.model_path)
    trainer.model.eval()

    # Use real sample data for export (not random - embedding lookup needs valid indices)
    feature_columns = feature_info['sparse_features'] + feature_info['dense_features']
    num_features = len(feature_columns)
    dummy_input = torch.from_numpy(train_df[feature_columns].iloc[:1].values.astype('float32'))

    print(f"Input shape: {dummy_input.shape}")
    print(f"Total features: {num_features}")

    # Determine output path
    if args.output_path is None:
        model_path = Path(args.model_path)
        args.output_path = str(model_path.parent / model_path.name.replace('.pth', '.onnx'))

    # Export to ONNX
    print(f"\n=== Exporting to ONNX ===")
    print(f"Output: {args.output_path}")

    # Enable dynamic batch size for efficient batched inference
    torch.onnx.export(
        trainer.model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy exporter for better compatibility
    )

    print("\n ONNX export successful!")
    print(f"ONNX model saved to: {args.output_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)
    print(" ONNX model verification passed")

    # Print model info
    file_size_mb = Path(args.output_path).stat().st_size / (1024 * 1024)
    print(f"\nModel size: {file_size_mb:.2f} MB")

    # Compare with PyTorch output using real data
    print("\n=== Verifying ONNX Output ===")
    import onnxruntime as ort
    import numpy as np

    # Use real sample data instead of random (single sample to match export shape)
    feature_columns = feature_info['sparse_features'] + feature_info['dense_features']
    sample_input = train_df[feature_columns].iloc[:1].values.astype(np.float32)
    sample_tensor = torch.from_numpy(sample_input)

    # PyTorch inference
    with torch.no_grad():
        torch_output = trainer.model(sample_tensor).numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(args.output_path)
    ort_output = ort_session.run(None, {'input': sample_input})[0]

    # Compare outputs
    max_diff = np.max(np.abs(torch_output - ort_output))
    print(f"Sample size: {len(sample_input)}")
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print(" ONNX output matches PyTorch output!")
    else:
        print(f" Warning: Difference is larger than expected (> 1e-5)")


if __name__ == '__main__':
    main()
