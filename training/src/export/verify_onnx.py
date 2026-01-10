"""
ONNX Verification Script

PyTorch 모델과 ONNX 모델의 출력을 비교하여 정합성을 검증합니다.

Usage:
    python training/src/export/verify_onnx.py --model-path training/models/deepfm_emb8_lr0.0005_best.pth
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.models.deepfm_trainer import DeepFMTrainer, CTRDataset


def load_processed_data(campaign_id: int,
                       data_base_dir: str = "training/data/processed",
                       neg_pos_ratio: int = None,
                       num_samples: int = 1000):
    """전처리된 데이터 로드"""
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

    # Load full train data for vocab size calculation
    train_df = pd.read_csv(data_path / "train.csv")
    # Also load val to ensure correct vocab sizes (important for downsampled data)
    val_df = pd.read_csv(data_path / "val.csv")

    # Load limited test samples for verification
    test_df = pd.read_csv(data_path / "test.csv", nrows=num_samples)

    with open(data_path / "feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    return train_df, val_df, test_df, feature_info


def main():
    parser = argparse.ArgumentParser(description='Verify ONNX model against PyTorch')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--onnx-path', type=str, default=None,
                       help='Path to ONNX model (default: same dir as PyTorch model)')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--output', type=str, default='training/results/onnx_verification.json',
                       help='Output verification results')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to test')
    parser.add_argument('--neg-pos-ratio', type=int, default=None,
                       help='Negative to positive ratio for downsampled data (e.g., 100 for 100:1). '
                            'If not set, uses original distribution.')
    parser.add_argument('--dnn-hidden', type=str, default='256,128,64',
                       help='DNN hidden units (comma-separated)')

    args = parser.parse_args()

    # Auto-detect hyperparameters
    parsed_params = DeepFMTrainer.parse_model_filename(args.model_path)
    embedding_dim = parsed_params['embedding_dim']

    # Parse DNN hidden units
    dnn_hidden_units = tuple(map(int, args.dnn_hidden.split(',')))

    # Determine ONNX path
    if args.onnx_path is None:
        model_path = Path(args.model_path)
        args.onnx_path = str(model_path.parent / model_path.name.replace('.pth', '.onnx'))

    print("=" * 60)
    print("ONNX Verification")
    print("=" * 60)
    print(f"PyTorch model: {args.model_path}")
    print(f"ONNX model: {args.onnx_path}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"DNN hidden: {dnn_hidden_units}")
    print(f"Test samples: {args.num_samples}")
    print("=" * 60)

    # Load data
    print("\n=== Loading Data ===")
    train_df, val_df, test_df, feature_info = load_processed_data(
        campaign_id=args.campaign,
        data_base_dir=args.data_base_dir,
        neg_pos_ratio=args.neg_pos_ratio,
        num_samples=args.num_samples
    )
    print(f"Loaded {len(train_df)} train samples (for vocab)")
    print(f"Loaded {len(val_df)} val samples (for vocab)")
    print(f"Loaded {len(test_df)} test samples (for verification)")

    # Load PyTorch model
    print("\n=== Loading PyTorch Model ===")
    trainer = DeepFMTrainer(
        feature_info=feature_info,
        embedding_dim=embedding_dim,
        dnn_hidden_units=dnn_hidden_units,
        device='cpu'
    )
    trainer.build_model(train_df, val_df)  # Pass val_df for correct vocab size
    trainer.load_best_model(args.model_path)
    trainer.model.eval()

    # Load ONNX model
    print("\n=== Loading ONNX Model ===")
    ort_session = ort.InferenceSession(args.onnx_path)
    print(f"ONNX Runtime providers: {ort_session.get_providers()}")

    # Prepare test data
    dataset = CTRDataset(test_df, feature_info)

    # Run inference
    print("\n=== Running Inference ===")
    pytorch_outputs = []
    onnx_outputs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0)  # Add batch dimension

            # PyTorch inference
            pytorch_out = trainer.model(x).squeeze().item()
            pytorch_outputs.append(pytorch_out)

            # ONNX inference
            ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
            ort_out = ort_session.run(None, ort_inputs)[0][0][0]
            onnx_outputs.append(ort_out)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples...")

    # Compare outputs
    print("\n=== Comparing Outputs ===")
    pytorch_outputs = np.array(pytorch_outputs)
    onnx_outputs = np.array(onnx_outputs)

    abs_diff = np.abs(pytorch_outputs - onnx_outputs)
    rel_diff = abs_diff / (np.abs(pytorch_outputs) + 1e-8)

    results = {
        'num_samples': len(dataset),
        'max_abs_diff': float(abs_diff.max()),
        'mean_abs_diff': float(abs_diff.mean()),
        'max_rel_diff': float(rel_diff.max()),
        'mean_rel_diff': float(rel_diff.mean()),
        'all_close_1e5': bool(np.allclose(pytorch_outputs, onnx_outputs, atol=1e-5)),
        'all_close_1e4': bool(np.allclose(pytorch_outputs, onnx_outputs, atol=1e-4)),
        'all_close_1e3': bool(np.allclose(pytorch_outputs, onnx_outputs, atol=1e-3)),
    }

    # Print results
    print(f"Samples tested: {results['num_samples']}")
    print(f"Max absolute difference: {results['max_abs_diff']:.2e}")
    print(f"Mean absolute difference: {results['mean_abs_diff']:.2e}")
    print(f"Max relative difference: {results['max_rel_diff']:.2%}")
    print(f"Mean relative difference: {results['mean_rel_diff']:.2%}")
    print(f"\nAll close (atol=1e-5): {results['all_close_1e5']}")
    print(f"All close (atol=1e-4): {results['all_close_1e4']}")
    print(f"All close (atol=1e-3): {results['all_close_1e3']}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Final verdict
    if results['all_close_1e5']:
        print("\n✅ VERIFICATION PASSED: PyTorch and ONNX outputs match within 1e-5 tolerance")
    elif results['all_close_1e4']:
        print("\n⚠️  VERIFICATION WARNING: Outputs match within 1e-4 but not 1e-5")
    else:
        print("\n❌ VERIFICATION FAILED: Outputs differ significantly")


if __name__ == '__main__':
    main()
