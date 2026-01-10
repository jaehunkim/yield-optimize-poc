"""
데이터 파이프라인 검증 스크립트

전체 파이프라인 테스트:
1. Raw 데이터 로딩 (iPinYou)
2. Feature 추출 및 전처리
3. Train/Val/Test 분할
4. Logistic Regression baseline 학습
5. AUC 측정 및 검증
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.src.data.load_data import iPinYouDataLoader


def load_and_process_data(campaign_id: int, days: int, output_base_dir: str):
    """데이터 로딩 및 전처리"""
    print(f"\n=== Loading Campaign {campaign_id} ({days} days) ===")

    loader = iPinYouDataLoader(campaign_id=campaign_id)

    # Get days
    imp_files = sorted(Path(loader.data_dir).glob(f"{loader.season}/imp.*.txt.bz2"))
    day_list = [f.stem.split('.')[1] for f in imp_files[:days]]

    print(f"Days: {day_list}")

    # Load impressions
    data = loader.load_campaign_data(day_list)

    # Add click labels
    data = loader.add_click_labels(data, day_list)

    # Prepare features
    data, feature_info = loader.prepare_features(data)

    # Save processed data
    output_path = loader.save_processed_data(
        data, feature_info,
        campaign_id=campaign_id,
        num_days=days,
        output_base_dir=output_base_dir
    )

    return output_path


def load_processed_data(data_dir: str = "training/data/processed"):
    """전처리된 데이터 로드"""
    data_path = Path(data_dir)

    print("\n=== Loading Processed Data ===")
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    with open(data_path / "feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_info['sparse_features'])} sparse + {len(feature_info['dense_features'])} dense")

    return train_df, val_df, test_df, feature_info


def check_data_quality(train_df, val_df, test_df, feature_info):
    """데이터 품질 체크"""
    print("\n=== Data Quality Check ===")

    all_features = feature_info['sparse_features'] + feature_info['dense_features']

    # Check for NaN
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nan_counts = df[all_features].isna().sum()
        if nan_counts.sum() > 0:
            print(f"WARNING: {name} has NaN values:")
            print(nan_counts[nan_counts > 0])
        else:
            print(f"{name}: No NaN values ✓")

    # Check feature ranges
    print("\nFeature ranges (train set, first 5):")
    for feat in all_features[:5]:
        min_val = train_df[feat].min()
        max_val = train_df[feat].max()
        print(f"  {feat}: [{min_val:.2f}, {max_val:.2f}]")

    # Check timestamp ordering
    print("\nTimestamp ordering check:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        is_sorted = df['timestamp'].is_monotonic_increasing
        print(f"  {name}: {'✓ Sorted' if is_sorted else '✗ NOT SORTED'}")


def prepare_onehot_data(train_df, val_df, test_df, feature_info):
    """One-Hot Encoding을 적용한 데이터 준비 (sparse matrix)"""
    sparse_features = feature_info['sparse_features']
    dense_features = feature_info['dense_features']

    # Sparse features: One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_train_sparse = encoder.fit_transform(train_df[sparse_features])
    X_val_sparse = encoder.transform(val_df[sparse_features])
    X_test_sparse = encoder.transform(test_df[sparse_features])

    # Dense features: as-is (already normalized)
    X_train_dense = train_df[dense_features].values
    X_val_dense = val_df[dense_features].values
    X_test_dense = test_df[dense_features].values

    # Combine sparse + dense
    X_train = hstack([X_train_sparse, X_train_dense])
    X_val = hstack([X_val_sparse, X_val_dense])
    X_test = hstack([X_test_sparse, X_test_dense])

    # Labels
    y_train = train_df[feature_info['target']].values
    y_val = val_df[feature_info['target']].values
    y_test = test_df[feature_info['target']].values

    print(f"One-Hot shape: {X_train_sparse.shape} ({X_train_sparse.shape[1]} categories)")
    print(f"Dense shape: {X_train_dense.shape}")
    print(f"Final shape: {X_train.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline(train_df, val_df, test_df, feature_info):
    """Logistic Regression baseline 학습 (One-Hot Encoding)"""
    print("\n=== Training Logistic Regression Baseline (One-Hot Encoding) ===")

    # Prepare data with One-Hot Encoding
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_onehot_data(
        train_df, val_df, test_df, feature_info
    )

    print(f"y_train distribution: {np.bincount(y_train.astype(int))}")
    print(f"Click rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

    # Train LR (supports sparse matrices)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=500,              # 피처가 많아졌으므로 반복 횟수 증가
        class_weight='balanced',
        solver='liblinear',        # 데이터셋이 아주 크지 않다면 liblinear가 더 안정적일 수 있음
        C=1.0,                     # 규제 강도 (너무 낮으면 0으로 수렴하므로 확인 필요)
        verbose=1,
        random_state=42
        # n_jobs=-1은 liblinear에서 지원하지 않으므로 주의
    )
    lr.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluation Results (One-Hot Encoding) ===")
    for name, X, y in [("Train", X_train, y_train),
                       ("Val", X_val, y_val),
                       ("Test", X_test, y_test)]:
        y_pred_proba = lr.predict_proba(X)[:, 1]
        y_pred = lr.predict(X)

        auc = roc_auc_score(y, y_pred_proba)
        logloss = log_loss(y, y_pred_proba)
        acc = accuracy_score(y, y_pred)

        print(f"{name:5s} - AUC: {auc:.4f}, LogLoss: {logloss:.4f}, Acc: {acc:.4f}")

    return lr


def main():
    parser = argparse.ArgumentParser(description='Load data and validate pipeline with LR baseline')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID (default: 2259)')
    parser.add_argument('--days', type=int, default=3,
                       help='Number of days to load (default: 3)')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload data even if cached')
    args = parser.parse_args()

    # Step 1: Determine data directory
    data_dir = Path(args.data_base_dir) / f"campaign_{args.campaign}" / f"{args.days}days"

    # Check if data already exists
    cache_exists = (data_dir / "train.csv").exists() and \
                   (data_dir / "val.csv").exists() and \
                   (data_dir / "test.csv").exists() and \
                   (data_dir / "feature_info.pkl").exists()

    if cache_exists and not args.force_reload:
        print(f"Using cached processed data in {data_dir}")
        print("(Use --force-reload to reload from raw data)")
    else:
        if args.force_reload:
            print("Force reloading data from raw files...")
        output_path = load_and_process_data(args.campaign, args.days, args.data_base_dir)
        data_dir = Path(output_path)

    # Step 2: Load processed data
    train_df, val_df, test_df, feature_info = load_processed_data(str(data_dir))

    # Step 3: Check data quality
    check_data_quality(train_df, val_df, test_df, feature_info)

    # Step 4: Train baseline
    lr_model = train_baseline(train_df, val_df, test_df, feature_info)

    print("\n=== Validation Complete ===")
    print("Data pipeline is ready for DeepFM training!")


if __name__ == '__main__':
    main()
