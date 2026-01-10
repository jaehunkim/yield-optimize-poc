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
        df = pd.read_csv(f, sep='\t', header=None, names=[
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

    def save_processed_data(self,
                           data: pd.DataFrame,
                           feature_info: Dict,
                           output_dir: str = "training/data/processed"):
        """
        전처리된 데이터 저장

        Args:
            data: Processed DataFrame
            feature_info: Feature information dictionary
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sort by timestamp to ensure chronological order
        data = data.sort_values('timestamp').reset_index(drop=True)

        # Split train/val/test chronologically (no data leakage)
        n = len(data)
        train_end = int(n * 0.72)
        val_end = int(n * 0.80)

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        print(f"Train period: {train_data['timestamp'].min()} ~ {train_data['timestamp'].max()}")
        print(f"Val period: {val_data['timestamp'].min()} ~ {val_data['timestamp'].max()}")
        print(f"Test period: {test_data['timestamp'].min()} ~ {test_data['timestamp'].max()}")

        # Save
        train_data.to_csv(output_path / "train.csv", index=False)
        val_data.to_csv(output_path / "val.csv", index=False)
        test_data.to_csv(output_path / "test.csv", index=False)

        # Save feature info
        with open(output_path / "feature_info.pkl", 'wb') as f:
            pickle.dump(feature_info, f)

        print(f"Saved processed data to {output_dir}")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Load iPinYou dataset')
    parser.add_argument('--campaign', type=int, default=2259,
                       help='Campaign ID (2259 or 3386)')
    parser.add_argument('--days', type=int, default=3,
                       help='Number of days to load (default: 3)')
    parser.add_argument('--output-dir', type=str, default='training/data/processed',
                       help='Output directory for processed data')

    args = parser.parse_args()

    # Load data
    loader = iPinYouDataLoader(campaign_id=args.campaign)

    # Use first N days
    imp_files = sorted(Path(loader.data_dir).glob(f"{loader.season}/imp.*.txt.bz2"))
    days = [f.stem.split('.')[1] for f in imp_files[:args.days]]

    print(f"Loading {len(days)} days: {days}")

    # Load impressions
    data = loader.load_campaign_data(days)

    # Add click labels
    data = loader.add_click_labels(data, days)

    # Prepare features
    data, feature_info = loader.prepare_features(data)

    # Save
    loader.save_processed_data(data, feature_info, args.output_dir)

    print("Done!")


if __name__ == '__main__':
    main()
