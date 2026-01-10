"""
iPinYou 데이터셋 로더

이 스크립트는:
1. iPinYou raw data (impression logs)를 읽어서
2. CTR 예측을 위한 feature를 추출하고
3. DeepCTR-PyTorch 형식으로 변환합니다
"""

import bz2
import os
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _load_single_day(args):
    """Helper function for parallel loading of impression files"""
    filepath, campaign_id = args

    with bz2.open(filepath, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None,
                        low_memory=False,  # Suppress dtype warning
                        names=[
            'bid_id', 'timestamp', 'log_type', 'ipinyou_id', 'user_agent',
            'ip', 'region', 'city', 'ad_exchange', 'domain', 'url',
            'anon_url_id', 'ad_slot_id', 'ad_slot_width', 'ad_slot_height',
            'ad_slot_visibility', 'ad_slot_format', 'ad_slot_floor_price',
            'creative_id', 'bidding_price', 'paying_price', 'key_page_url',
            'advertiser_id', 'user_tags'
        ])
        # Filter by campaign
        return df[df['advertiser_id'] == campaign_id]


def _load_click_file(filepath):
    """Helper function for parallel loading of click files"""
    click_ids = set()
    with bz2.open(filepath, 'rt') as f:
        for line in f:
            bid_id = line.split('\t')[0]
            click_ids.add(bid_id)
    return click_ids


class iPinYouDataLoader:
    """iPinYou 데이터셋 로더"""

    def __init__(self,
                 data_dir: str = "training/data/raw/ipinyou.contest.dataset",
                 campaign_id: int = 2259,
                 n_jobs: int = -1):
        """
        Args:
            data_dir: iPinYou 데이터 디렉토리
            campaign_id: 사용할 캠페인 ID (2259, 3386 등)
            n_jobs: 병렬 처리에 사용할 프로세스 수 (-1이면 CPU 코어 수만큼)
        """
        self.data_dir = Path(data_dir)
        self.campaign_id = campaign_id
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Campaign에 따라 season 자동 선택
        # training2nd: 1458 (easy, AUC 0.98), 3358, 3386 (hard, AUC 0.74-0.84), 3427, 3476
        # training3rd: 2259 (medium, AUC 0.65-0.72), 2261, 2821, 2997
        if campaign_id in [1458, 3358, 3386, 3427, 3476]:
            self.season = "training2nd"
        elif campaign_id in [2259, 2261, 2821, 2997]:
            self.season = "training3rd"
        else:
            # 알 수 없는 캠페인
            raise ValueError(f"Unknown campaign_id: {campaign_id}. "
                           f"Available: training2nd={[1458, 3358, 3386, 3427, 3476]}, "
                           f"training3rd={[2259, 2261, 2821, 2997]}")

        # Feature columns from iPinYou dataset
        # Format: [bid_id, timestamp, log_type, iPinYou_id, user_agent, IP, region, city,
        #          ad_exchange, domain, url, anonymous_url_id, ad_slot_id, ad_slot_width,
        #          ad_slot_height, ad_slot_visibility, ad_slot_format, ad_slot_floor_price,
        #          creative_id, bidding_price, paying_price, key_page_url, advertiser_id,
        #          user_tags]

    def load_campaign_data(self, days: List[str] = None) -> pd.DataFrame:
        """
        특정 날짜들의 impression 데이터를 로드 (병렬 처리)

        Args:
            days: 날짜 리스트 (e.g., ['20131019', '20131020'])
                  None이면 모든 날짜 로드

        Returns:
            DataFrame with impression data
        """
        if days is None:
            # Get all available days
            imp_files = sorted(self.data_dir.glob(f"{self.season}/imp.*.txt.bz2"))
            days = [f.stem.split('.')[1] for f in imp_files]

        # Prepare file paths
        filepaths = []
        for day in days:
            filepath = self.data_dir / self.season / f"imp.{day}.txt.bz2"
            if filepath.exists():
                filepaths.append((filepath, self.campaign_id))
            else:
                print(f"Warning: {filepath} not found, skipping")

        # Parallel loading
        print(f"Loading {len(filepaths)} days using {self.n_jobs} processes...")
        with Pool(self.n_jobs) as pool:
            all_data = list(tqdm(
                pool.imap(_load_single_day, filepaths),
                total=len(filepaths),
                desc="Loading days"
            ))

        data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(data)} impressions from {len(days)} days (Campaign {self.campaign_id})")

        return data

    def add_click_labels(self, imp_data: pd.DataFrame, days: List[str]) -> pd.DataFrame:
        """
        클릭 로그를 읽어서 impression에 click label 추가 (병렬 처리)

        Args:
            imp_data: impression DataFrame
            days: 날짜 리스트

        Returns:
            DataFrame with 'click' column (1 if clicked, 0 otherwise)
        """
        # Prepare click file paths
        click_files = []
        for day in days:
            filepath = self.data_dir / self.season / f"clk.{day}.txt.bz2"
            if filepath.exists():
                click_files.append(filepath)

        # Parallel loading
        print(f"Loading click files using {self.n_jobs} processes...")
        with Pool(self.n_jobs) as pool:
            click_id_sets = list(tqdm(
                pool.imap(_load_click_file, click_files),
                total=len(click_files),
                desc="Loading clicks"
            ))

        # Merge all click IDs
        click_bid_ids = set()
        for click_ids in click_id_sets:
            click_bid_ids.update(click_ids)

        # Add click label
        imp_data['click'] = imp_data['bid_id'].isin(click_bid_ids).astype(int)

        click_rate = imp_data['click'].mean()
        print(f"Click rate: {click_rate:.4f} ({imp_data['click'].sum()} clicks / {len(imp_data)} impressions)")

        return imp_data

    def load_testing_data(self) -> pd.DataFrame:
        """
        Load official iPinYou testing/leaderboard data

        Testing data format (26 fields):
        - First 24 fields: same as impression log
        - Field 25: number of clicks (C) - 0, 1, or more
        - Field 26: has conversion (V) - 0 or 1

        Returns:
            DataFrame with testing data including click labels
        """
        # Determine testing file based on season
        testing_season = self.season.replace('training', 'testing')

        # Find testing file
        testing_files = list(self.data_dir.glob(f"{testing_season}/leaderboard.test.data.*.txt.bz2"))

        if not testing_files:
            raise FileNotFoundError(
                f"No testing data found in {self.data_dir / testing_season}/"
            )

        testing_file = testing_files[0]
        print(f"Loading testing data from {testing_file.name}...")

        # Load testing data (26 fields)
        with bz2.open(testing_file, 'rt') as f:
            df = pd.read_csv(f, sep='\t', header=None,
                           low_memory=False,  # Suppress dtype warning
                           names=[
                'bid_id', 'timestamp', 'log_type', 'ipinyou_id', 'user_agent',
                'ip', 'region', 'city', 'ad_exchange', 'domain', 'url',
                'anon_url_id', 'ad_slot_id', 'ad_slot_width', 'ad_slot_height',
                'ad_slot_visibility', 'ad_slot_format', 'ad_slot_floor_price',
                'creative_id', 'bidding_price', 'paying_price', 'key_page_url',
                'advertiser_id', 'user_tags', 'click_count', 'has_conversion'
            ])

        # Filter by campaign
        df = df[df['advertiser_id'] == self.campaign_id].copy()

        # Convert click_count to binary click label (1 if any clicks, 0 otherwise)
        df['click'] = (df['click_count'] > 0).astype(int)

        # Drop the extra columns we don't need for CTR prediction
        df = df.drop(columns=['click_count', 'has_conversion'])

        click_rate = df['click'].mean()
        print(f"Loaded {len(df)} testing impressions (Campaign {self.campaign_id})")
        print(f"Testing click rate: {click_rate:.4f} ({df['click'].sum()} clicks)")

        return df

    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        DeepFM을 위한 feature 준비

        Args:
            data: Raw DataFrame

        Returns:
            (processed_data, feature_info)
        """
        # Select useful categorical features
        cat_features = [
            'region', 'city', 'ad_exchange', 'domain',
            'ad_slot_id', 'ad_slot_width', 'ad_slot_height',
            'ad_slot_visibility', 'ad_slot_format',
            'creative_id', 'advertiser_id'
        ]

        # Select numerical features
        num_features = [
            'ad_slot_floor_price', 'bidding_price', 'paying_price'
        ]

        # Handle user_tags (sparse feature)
        # user_tags format: "tag1,tag2,tag3" - we'll use first tag only for simplicity
        data['user_tag'] = data['user_tags'].fillna('').str.split(',').str[0]
        cat_features.append('user_tag')

        # Fill missing values
        for col in cat_features:
            data[col] = data[col].fillna('MISSING').astype(str)

        for col in num_features:
            data[col] = data[col].fillna(0).astype(float)

        # Build feature info for DeepCTR
        feature_info = {
            'sparse_features': cat_features,
            'dense_features': num_features,
            'target': 'click'
        }

        # Encode categorical features
        for feat in cat_features:
            data[feat] = pd.Categorical(data[feat]).codes

        # Normalize numerical features
        for feat in num_features:
            mean = data[feat].mean()
            std = data[feat].std()
            if std > 0:
                data[feat] = (data[feat] - mean) / std
            feature_info[f'{feat}_mean'] = mean
            feature_info[f'{feat}_std'] = std

        return data, feature_info

    @staticmethod
    def downsample_negatives(data: pd.DataFrame,
                            neg_pos_ratio: int = 100,
                            target_col: str = 'click',
                            random_seed: int = 42) -> pd.DataFrame:
        """
        Negative downsampling to handle class imbalance

        Args:
            data: DataFrame with click labels
            neg_pos_ratio: Ratio of negatives to positives (e.g., 100 means 100:1)
            target_col: Target column name
            random_seed: Random seed for reproducibility

        Returns:
            Downsampled DataFrame
        """
        import numpy as np

        # Separate positive and negative samples
        pos_data = data[data[target_col] == 1].copy()
        neg_data = data[data[target_col] == 0].copy()

        n_pos = len(pos_data)
        n_neg = len(neg_data)

        # Calculate number of negatives to sample
        n_neg_sampled = min(n_pos * neg_pos_ratio, n_neg)

        print(f"\n--- Negative Downsampling ---")
        print(f"Original: Pos={n_pos}, Neg={n_neg} (ratio 1:{n_neg/n_pos:.1f})")
        print(f"Target ratio: 1:{neg_pos_ratio}")

        # Sample negatives
        np.random.seed(random_seed)
        neg_sampled = neg_data.sample(n=n_neg_sampled, random_state=random_seed)

        # Combine and shuffle
        result = pd.concat([pos_data, neg_sampled], ignore_index=True)
        result = result.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        print(f"Sampled: Pos={n_pos}, Neg={n_neg_sampled} (ratio 1:{n_neg_sampled/n_pos:.1f})")
        print(f"Total samples: {len(result)} ({len(result)/len(data)*100:.1f}% of original)")
        print(f"New click rate: {result[target_col].mean():.4f}")

        return result

    def save_processed_data(self,
                           train_data: pd.DataFrame,
                           test_data: pd.DataFrame,
                           feature_info: Dict,
                           campaign_id: int,
                           output_base_dir: str = "training/data/processed",
                           val_split: float = 0.2,
                           neg_pos_ratio: int = None):
        """
        전처리된 데이터 저장 (캠페인별 디렉토리)

        Args:
            train_data: Processed training DataFrame (from training set)
            test_data: Processed testing DataFrame (from official testing set)
            feature_info: Feature information dictionary
            campaign_id: Campaign ID
            output_base_dir: Base output directory
            val_split: Validation split ratio from training data (default: 0.2)
            neg_pos_ratio: If set, apply negative downsampling (e.g., 100 for 100:1 ratio)
        """
        # Create campaign subdirectory with sampling ratio suffix
        if neg_pos_ratio is not None:
            dir_name = f"campaign_{campaign_id}_neg{neg_pos_ratio}"
        else:
            dir_name = f"campaign_{campaign_id}"

        output_path = Path(output_base_dir) / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Sort training data by timestamp to ensure chronological order
        train_data = train_data.sort_values('timestamp').reset_index(drop=True)

        # Split training data into train/val chronologically FIRST (no data leakage)
        n = len(train_data)
        train_end = int(n * (1 - val_split))

        final_train = train_data.iloc[:train_end].copy()
        final_val = train_data.iloc[train_end:].copy()

        # Apply negative downsampling ONLY to training set (NOT val/test)
        if neg_pos_ratio is not None:
            print(f"\n=== Applying Negative Downsampling to TRAINING SET ONLY ===")
            final_train = self.downsample_negatives(
                final_train,
                neg_pos_ratio=neg_pos_ratio,
                target_col='click'
            )
            print(f"Val/Test sets remain at original distribution")

        print(f"\n=== Data Split (Using Official Testing Set) ===")
        if neg_pos_ratio is not None:
            print(f"Negative Downsampling: 1:{neg_pos_ratio}")
        print(f"Train: {len(final_train)} ({len(final_train)/n*100:.1f}%)")
        print(f"Val: {len(final_val)} ({len(final_val)/n*100:.1f}%)")
        print(f"Test: {len(test_data)} (official leaderboard set)")
        print(f"\nTrain period: {final_train['timestamp'].min()} ~ {final_train['timestamp'].max()}")
        print(f"Val period: {final_val['timestamp'].min()} ~ {final_val['timestamp'].max()}")
        print(f"Test period: {test_data['timestamp'].min()} ~ {test_data['timestamp'].max()}")

        # Save
        final_train.to_csv(output_path / "train.csv", index=False)
        final_val.to_csv(output_path / "val.csv", index=False)
        test_data.to_csv(output_path / "test.csv", index=False)

        # Save feature info
        with open(output_path / "feature_info.pkl", 'wb') as f:
            pickle.dump(feature_info, f)

        print(f"\nSaved processed data to {output_path}")
        return str(output_path)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Load iPinYou dataset')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID (2259 or 3386)')
    parser.add_argument('--output-dir', type=str, default='training/data/processed',
                       help='Base output directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--neg-pos-ratio', type=int, default=None,
                       help='Negative to positive ratio for downsampling (e.g., 100 for 100:1). '
                            'If not set, no downsampling is applied.')

    args = parser.parse_args()

    # Check if data already exists
    if args.neg_pos_ratio is not None:
        expected_path = Path(args.output_dir) / f"campaign_{args.campaign}_neg{args.neg_pos_ratio}"
    else:
        expected_path = Path(args.output_dir) / f"campaign_{args.campaign}"

    if (expected_path / "train.csv").exists() and \
       (expected_path / "val.csv").exists() and \
       (expected_path / "test.csv").exists() and \
       (expected_path / "feature_info.pkl").exists():
        print(f"Data already exists at {expected_path}")
        print("Use --force to reload")
        return

    # Load data
    loader = iPinYouDataLoader(campaign_id=args.campaign)

    # Load ALL available training days for this campaign
    print(f"\n{'='*60}")
    print(f"Loading iPinYou Dataset - Campaign {args.campaign}")
    print(f"{'='*60}")

    # Load training impressions (all days)
    print(f"\n--- Loading Training Data ---")
    train_data = loader.load_campaign_data(days=None)  # None = load all days

    # Add click labels to training data
    imp_files = sorted(Path(loader.data_dir).glob(f"{loader.season}/imp.*.txt.bz2"))
    days = [f.stem.split('.')[1] for f in imp_files]
    train_data = loader.add_click_labels(train_data, days)

    # Prepare features for training data
    train_data, feature_info = loader.prepare_features(train_data)

    # Load official testing data
    print(f"\n--- Loading Official Testing Data ---")
    test_data = loader.load_testing_data()

    # Prepare features for testing data (using same feature_info for consistency)
    test_data, _ = loader.prepare_features(test_data)

    # Save train/val/test splits
    output_path = loader.save_processed_data(
        train_data=train_data,
        test_data=test_data,
        feature_info=feature_info,
        campaign_id=args.campaign,
        output_base_dir=args.output_dir,
        val_split=args.val_split,
        neg_pos_ratio=args.neg_pos_ratio
    )

    print(f"\n{'='*60}")
    print(f"Done! Data saved to {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
