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


def load_and_process_data(campaign_id: int, output_base_dir: str):
    """데이터 로딩 및 전처리"""
    print(f"\n=== Loading Campaign {campaign_id} (ALL training days) ===")

    loader = iPinYouDataLoader(campaign_id=campaign_id)

    # Load ALL training impressions
    train_data = loader.load_campaign_data(days=None)  # None = all days

    # Get all days for click labels
    imp_files = sorted(Path(loader.data_dir).glob(f"{loader.season}/imp.*.txt.bz2"))
    day_list = [f.stem.split('.')[1] for f in imp_files]

    print(f"Training days: {day_list}")

    # Add click labels
    train_data = loader.add_click_labels(train_data, day_list)

    # Prepare features
    train_data, feature_info = loader.prepare_features(train_data)

    # Load official testing data
    print(f"\n--- Loading Official Testing Data ---")
    test_data = loader.load_testing_data()
    test_data, _ = loader.prepare_features(test_data)

    # Save processed data
    output_path = loader.save_processed_data(
        train_data=train_data,
        test_data=test_data,
        feature_info=feature_info,
        campaign_id=campaign_id,
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


def add_temporal_features(df):
    """Add hour and day_of_week from timestamp"""
    # iPinYou timestamp format: YYYYMMDDHHMMSSmmm (e.g., 20131019000103546)
    # Extract hour and day_of_week directly from the string format

    df = df.copy()

    # Convert timestamp to string if not already
    df['timestamp_str'] = df['timestamp'].astype(str)

    # Extract hour (positions 8-10 in YYYYMMDDHHMMSSmmm)
    df['hour'] = df['timestamp_str'].str[8:10].astype(int)

    # Extract date for day_of_week (YYYYMMDD)
    df['date_str'] = df['timestamp_str'].str[:8]
    df['datetime'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Clean up temporary columns
    df = df.drop(columns=['timestamp_str', 'date_str', 'datetime'])

    return df


def target_encode_feature(train_df, val_df, test_df, feature, target='click', smoothing=100):
    """
    Target Encoding with Smoothing

    Args:
        train_df, val_df, test_df: DataFrames
        feature: Feature name to encode
        target: Target column name
        smoothing: Smoothing parameter (higher = more conservative)

    Returns:
        Encoded values for train, val, test
    """
    # Calculate global mean from training set
    global_mean = train_df[target].mean()

    # Calculate per-category statistics in training set
    agg = train_df.groupby(feature)[target].agg(['sum', 'count'])

    # Smoothed target encoding
    # Formula: (sum + smoothing * global_mean) / (count + smoothing)
    agg['encoded'] = (agg['sum'] + smoothing * global_mean) / (agg['count'] + smoothing)

    # Map to all datasets
    encoding_map = agg['encoded'].to_dict()

    train_encoded = train_df[feature].map(encoding_map).fillna(global_mean)
    val_encoded = val_df[feature].map(encoding_map).fillna(global_mean)
    test_encoded = test_df[feature].map(encoding_map).fillna(global_mean)

    return train_encoded, val_encoded, test_encoded


def prepare_improved_data(train_df, val_df, test_df, feature_info, min_count=10, smoothing=100):
    """
    Improved feature engineering:
    1. Target Encoding for high-cardinality features
    2. One-Hot Encoding for low-cardinality stable features
    3. Temporal features (hour, day_of_week)
    """
    print("\n=== Improved Feature Engineering ===")

    # Add temporal features
    print("\n--- Adding Temporal Features ---")
    train_df = add_temporal_features(train_df)
    val_df = add_temporal_features(val_df)
    test_df = add_temporal_features(test_df)

    sparse_features = feature_info['sparse_features']
    dense_features = feature_info['dense_features']

    # Categorize features
    # High-cardinality: use Target Encoding (domain, ad_slot_id, creative_id)
    # Low-cardinality: use One-Hot Encoding (region, city, ad_exchange, etc.)
    high_card_features = ['domain', 'ad_slot_id', 'creative_id', 'user_tag']
    low_card_features = [f for f in sparse_features if f not in high_card_features]

    print(f"\nHigh-cardinality features (Target Encoding): {high_card_features}")
    print(f"Low-cardinality features (One-Hot Encoding): {low_card_features}")

    # Apply Target Encoding for high-cardinality features
    print(f"\n--- Applying Target Encoding (smoothing={smoothing}) ---")
    target_encoded_features = []

    for feat in high_card_features:
        train_enc, val_enc, test_enc = target_encode_feature(
            train_df, val_df, test_df, feat, smoothing=smoothing
        )
        train_df[f'{feat}_encoded'] = train_enc
        val_df[f'{feat}_encoded'] = val_enc
        test_df[f'{feat}_encoded'] = test_enc
        target_encoded_features.append(f'{feat}_encoded')

        n_unique_train = train_df[feat].nunique()
        n_unique_test = test_df[feat].nunique()
        overlap = set(train_df[feat].unique()) & set(test_df[feat].unique())
        print(f"  {feat}: Train={n_unique_train}, Test={n_unique_test}, Overlap={len(overlap)} ({len(overlap)/n_unique_test*100:.1f}%)")

    # One-Hot Encoding for low-cardinality features (with min count filtering)
    print(f"\n--- Applying One-Hot Encoding (min_count={min_count}) ---")
    train_filtered = train_df.copy()
    val_filtered = val_df.copy()
    test_filtered = test_df.copy()

    for feat in low_card_features:
        value_counts = train_df[feat].value_counts()
        rare_values = value_counts[value_counts < min_count].index

        train_filtered[feat] = train_df[feat].apply(lambda x: -1 if x in rare_values else x)
        val_filtered[feat] = val_df[feat].apply(lambda x: -1 if x in rare_values else x)
        test_filtered[feat] = test_df[feat].apply(lambda x: -1 if x in rare_values else x)

        kept = len(value_counts) - len(rare_values)
        print(f"  {feat}: {len(value_counts)} -> {kept} categories")

    # One-Hot Encode low-cardinality features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_train_sparse = encoder.fit_transform(train_filtered[low_card_features])
    X_val_sparse = encoder.transform(val_filtered[low_card_features])
    X_test_sparse = encoder.transform(test_filtered[low_card_features])

    # Combine all features
    # 1. Target-encoded features
    # 2. Original dense features
    # 3. Temporal features (hour, day_of_week)
    # 4. One-Hot encoded sparse features

    target_encoded_cols = target_encoded_features
    temporal_cols = ['hour', 'day_of_week']
    all_dense_cols = target_encoded_cols + dense_features + temporal_cols

    X_train_dense = train_df[all_dense_cols].values
    X_val_dense = val_df[all_dense_cols].values
    X_test_dense = test_df[all_dense_cols].values

    # Combine (sparse OHE + dense features)
    X_train = hstack([X_train_sparse, X_train_dense]).tocsr()
    X_val = hstack([X_val_sparse, X_val_dense]).tocsr()
    X_test = hstack([X_test_sparse, X_test_dense]).tocsr()

    # Labels
    y_train = train_df[feature_info['target']].values
    y_val = val_df[feature_info['target']].values
    y_test = test_df[feature_info['target']].values

    print(f"\n--- Final Feature Dimensions ---")
    print(f"One-Hot sparse: {X_train_sparse.shape[1]} features")
    print(f"Target-encoded: {len(target_encoded_cols)} features")
    print(f"Original dense: {len(dense_features)} features")
    print(f"Temporal: {len(temporal_cols)} features")
    print(f"Total: {X_train.shape[1]} features")

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_onehot_data(train_df, val_df, test_df, feature_info, min_count=10):
    """One-Hot Encoding을 적용한 데이터 준비 (sparse matrix + min count filtering)"""
    sparse_features = feature_info['sparse_features']
    dense_features = feature_info['dense_features']

    # Step 1: Min count filtering for sparse features
    print(f"\n--- Applying Min Count Filtering (min_count={min_count}) ---")
    train_filtered = train_df.copy()
    val_filtered = val_df.copy()
    test_filtered = test_df.copy()

    for feat in sparse_features:
        # Count frequency in training set
        value_counts = train_df[feat].value_counts()
        rare_values = value_counts[value_counts < min_count].index

        # Replace rare values with -1 (unknown)
        train_filtered[feat] = train_df[feat].apply(lambda x: -1 if x in rare_values else x)
        val_filtered[feat] = val_df[feat].apply(lambda x: -1 if x in rare_values else x)
        test_filtered[feat] = test_df[feat].apply(lambda x: -1 if x in rare_values else x)

        kept = len(value_counts) - len(rare_values)
        print(f"  {feat}: {len(value_counts)} -> {kept} categories (removed {len(rare_values)})")

    # Step 2: Sparse features: One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_train_sparse = encoder.fit_transform(train_filtered[sparse_features])
    X_val_sparse = encoder.transform(val_filtered[sparse_features])
    X_test_sparse = encoder.transform(test_filtered[sparse_features])

    # Dense features: as-is (already normalized)
    X_train_dense = train_df[dense_features].values
    X_val_dense = val_df[dense_features].values
    X_test_dense = test_df[dense_features].values

    # Combine sparse + dense (convert to CSR for efficient indexing)
    X_train = hstack([X_train_sparse, X_train_dense]).tocsr()
    X_val = hstack([X_val_sparse, X_val_dense]).tocsr()
    X_test = hstack([X_test_sparse, X_test_dense]).tocsr()

    # Labels
    y_train = train_df[feature_info['target']].values
    y_val = val_df[feature_info['target']].values
    y_test = test_df[feature_info['target']].values

    print(f"\nOne-Hot shape: {X_train_sparse.shape} ({X_train_sparse.shape[1]} categories)")
    print(f"Dense shape: {X_train_dense.shape}")
    print(f"Final shape: {X_train.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def downsample_negatives(X_train, y_train, neg_sample_ratio=0.1):
    """Negative downsampling to balance the dataset"""
    from scipy.sparse import issparse

    print(f"\n--- Applying Negative Downsampling (ratio={neg_sample_ratio}) ---")

    # Find positive and negative indices
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # Sample negatives
    n_neg_samples = int(len(neg_idx) * neg_sample_ratio)
    np.random.seed(42)
    sampled_neg_idx = np.random.choice(neg_idx, size=n_neg_samples, replace=False)

    # Combine positive and sampled negatives
    combined_idx = np.concatenate([pos_idx, sampled_neg_idx])
    np.random.shuffle(combined_idx)

    # Handle sparse matrix indexing (CSR supports fancy indexing)
    if issparse(X_train):
        X_train_sampled = X_train[combined_idx]
    else:
        X_train_sampled = X_train[combined_idx]

    y_train_sampled = y_train[combined_idx]

    print(f"Original: {len(y_train)} samples (Pos: {len(pos_idx)}, Neg: {len(neg_idx)})")
    print(f"Sampled: {len(y_train_sampled)} samples (Pos: {len(pos_idx)}, Neg: {n_neg_samples})")
    print(f"New click rate: {y_train_sampled.mean():.4f}")

    return X_train_sampled, y_train_sampled


def train_baseline(train_df, val_df, test_df, feature_info,
                   min_count=10, use_downsampling=True, neg_sample_ratio=0.1):
    """Logistic Regression baseline 학습 (One-Hot Encoding + Improvements)"""
    print("\n=== Training Logistic Regression Baseline (One-Hot Only) ===")

    # Prepare data with One-Hot Encoding + Min Count Filtering
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_onehot_data(
        train_df, val_df, test_df, feature_info, min_count=min_count
    )

    print(f"\nOriginal data distribution:")
    print(f"  y_train: {np.bincount(y_train.astype(int))}")
    print(f"  Click rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

    # Apply negative downsampling
    if use_downsampling:
        X_train, y_train = downsample_negatives(X_train, y_train, neg_sample_ratio)

    # Train LR with stronger regularization
    print("\nTraining Logistic Regression with L1 regularization...")
    lr = LogisticRegression(
        penalty='l1',              # L1 for feature selection
        C=0.1,                     # Stronger regularization
        solver='liblinear',
        max_iter=500,
        class_weight={0: 1, 1: 10},  # Manual class weighting
        verbose=1,
        random_state=42
    )
    lr.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluation Results (One-Hot Model) ===")
    for name, X, y in [("Train", X_train, y_train),
                       ("Val", X_val, y_val),
                       ("Test", X_test, y_test)]:
        y_pred_proba = lr.predict_proba(X)[:, 1]
        y_pred = lr.predict(X)

        auc = roc_auc_score(y, y_pred_proba)
        logloss = log_loss(y, y_pred_proba)
        acc = accuracy_score(y, y_pred)

        print(f"{name:5s} - AUC: {auc:.4f}, LogLoss: {logloss:.4f}, Acc: {acc:.4f}")

    # Count non-zero coefficients (feature selection effect)
    n_features_used = np.sum(lr.coef_ != 0)
    print(f"\nL1 Feature Selection: {n_features_used} / {lr.coef_.shape[1]} features used")

    return lr


def train_improved_baseline(train_df, val_df, test_df, feature_info,
                            min_count=10, smoothing=100,
                            use_downsampling=True, neg_sample_ratio=0.1):
    """
    Improved Logistic Regression baseline:
    - Target Encoding for high-cardinality features
    - Temporal features (hour, day_of_week)
    - One-Hot Encoding for stable low-cardinality features
    """
    print("\n=== Training IMPROVED Logistic Regression (Target Encoding + Temporal) ===")

    # Prepare data with improved feature engineering
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_improved_data(
        train_df, val_df, test_df, feature_info, min_count=min_count, smoothing=smoothing
    )

    print(f"\nOriginal data distribution:")
    print(f"  y_train: {np.bincount(y_train.astype(int))}")
    print(f"  Click rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")

    # Apply negative downsampling
    if use_downsampling:
        X_train, y_train = downsample_negatives(X_train, y_train, neg_sample_ratio)

    # Train LR with L1 regularization
    print("\nTraining Logistic Regression with L1 regularization...")
    lr = LogisticRegression(
        penalty='l1',
        C=0.01,                    # Even stronger regularization for robustness
        solver='liblinear',
        max_iter=500,
        class_weight={0: 1, 1: 10},
        verbose=1,
        random_state=42
    )
    lr.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluation Results (Improved Model with Target Encoding) ===")
    for name, X, y in [("Train", X_train, y_train),
                       ("Val", X_val, y_val),
                       ("Test", X_test, y_test)]:
        y_pred_proba = lr.predict_proba(X)[:, 1]
        y_pred = lr.predict(X)

        auc = roc_auc_score(y, y_pred_proba)
        logloss = log_loss(y, y_pred_proba)
        acc = accuracy_score(y, y_pred)

        print(f"{name:5s} - AUC: {auc:.4f}, LogLoss: {logloss:.4f}, Acc: {acc:.4f}")

    # Count non-zero coefficients
    n_features_used = np.sum(lr.coef_ != 0)
    print(f"\nL1 Feature Selection: {n_features_used} / {lr.coef_.shape[1]} features used")

    return lr


def main():
    parser = argparse.ArgumentParser(description='Load data and validate pipeline with LR baseline')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID (default: 2259)')
    parser.add_argument('--data-base-dir', type=str, default='training/data/processed',
                       help='Base processed data directory')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload data even if cached')
    args = parser.parse_args()

    # Step 1: Determine data directory
    data_dir = Path(args.data_base_dir) / f"campaign_{args.campaign}"

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
        output_path = load_and_process_data(args.campaign, args.data_base_dir)
        data_dir = Path(output_path)

    # Step 2: Load processed data
    train_df, val_df, test_df, feature_info = load_processed_data(str(data_dir))

    # Step 3: Check data quality
    check_data_quality(train_df, val_df, test_df, feature_info)

    # Step 4a: Train original baseline (One-Hot only)
    # print("\n" + "="*80)
    # print("BASELINE 1: One-Hot Encoding Only")
    # print("="*80)
    # lr_model_ohe = train_baseline(train_df, val_df, test_df, feature_info)

    # Step 4b: Train improved baseline (Target Encoding + Temporal)
    print("\n" + "="*80)
    print("BASELINE 2: Target Encoding + Temporal Features")
    print("="*80)
    lr_model_improved = train_improved_baseline(train_df, val_df, test_df, feature_info)

    print("\n" + "="*80)
    print("=== Validation Complete ===")
    print("="*80)
    print("\nCompare the two baselines:")
    print("  1. One-Hot only: Should show high Val AUC but poor Test AUC (overfitting to IDs)")
    print("  2. Target Encoding + Temporal: Should show more balanced Val/Test AUC")
    print("\nData pipeline is ready for DeepFM training!")


if __name__ == '__main__':
    main()
