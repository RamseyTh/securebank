import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

class Feature_Extractor:
    def extract(self, training_dataset_filename: str, testing_dataset_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        current_dir = os.getcwd()
        file_dir = os.path.join(os.path.dirname(current_dir), 'storage/partitioned_data')

        train_data = pd.read_parquet(f"{file_dir}/{training_dataset_filename}")
        test_data = pd.read_parquet(f"{file_dir}/{testing_dataset_filename}")
        return train_data, test_data

    def transform(self, training_dataset: pd.DataFrame, testing_dataset: pd.DataFrame) -> List[pd.DataFrame]:
        def extract_features(df):
            # Time-based features
            df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
            df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
            df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
            df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            
            # Transaction amount features
            df['log_amt'] = np.log1p(df['amt'])
            
            # Merchant category features
            df['category'] = pd.Categorical(df['category']).codes
            
            # Location-based features
            df['distance'] = self.haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
            
            # Select final features
            features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'log_amt', 'category', 'distance']
            target = 'is_fraud'
            
            return df[features], df[target]

        X_train, y_train = extract_features(training_dataset)
        X_test, y_test = extract_features(testing_dataset)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return [pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train,
                pd.DataFrame(X_test_scaled, columns=X_test.columns), y_test]

    def describe(self, partitioned_data: List[pd.DataFrame]) -> Dict:
        description = {
            'version': 'v1.0',
            'storage': 'securebank/storage/features/',
            'description': {}
        }
        
        for i, dataset in enumerate(['train_features', 'train_target', 'test_features', 'test_target']):
            description['description'][dataset] = {
                'shape': partitioned_data[i].shape,
                'columns': partitioned_data[i].columns.tolist() if isinstance(partitioned_data[i], pd.DataFrame) else None,
                'dtypes': partitioned_data[i].dtypes.to_dict() if isinstance(partitioned_data[i], pd.DataFrame) else None
            }
        
        return description

    def load(self, partitioned_data: List[pd.DataFrame], output_filename: str) -> None:
        current_dir = os.getcwd()
        save_to_dir = os.path.join(os.path.dirname(current_dir), 'storage/features')
        
        for i, dataset in enumerate(['train_features', 'train_target', 'test_features', 'test_target']):
            partitioned_data[i].to_parquet(f"{save_to_dir}/{dataset}_{output_filename}")

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c

        return distance