import os
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import train_test_split

class Dataset_Designer:
    def extract(self, raw_dataset_filename: str) -> pd.DataFrame:
        current_dir = os.getcwd()
        file_dir = os.path.join(os.path.dirname(current_dir), 'storage/raw_data')
        file_path = os.path.join(file_dir, raw_dataset_filename)

        return pd.read_parquet(file_path)

    def sample(self, raw_dataset: pd.DataFrame) -> List[pd.DataFrame]:
        train_data, test_data = train_test_split(raw_dataset, test_size=0.2, random_state=42, stratify=raw_dataset['is_fraud'])

        return [train_data, test_data]

    def describe(self, partitioned_data: List[pd.DataFrame]) -> Dict:
        description = {
            'version': 'v1.0',
            'storage': 'securebank/storage/partitioned_data/',
            'description': {}
        }
        
        for i, dataset in enumerate(['train', 'test']):
            description['description'][dataset] = {
                'data_type': dataset,
                'shape': partitioned_data[i].shape,
                'fraud_ratio': partitioned_data[i]['is_fraud'].mean()
            }
        
        return description

    def load(self, partitioned_data: List[pd.DataFrame], output_filename: str) -> None:
        current_dir = os.getcwd()
        save_to_dir = os.path.join(os.path.dirname(current_dir), 'storage/partitioned_data')
        
        for i, dataset in enumerate(['train', 'test']):
            partitioned_data[i].to_parquet(f"{save_to_dir}/{dataset}_{output_filename}")