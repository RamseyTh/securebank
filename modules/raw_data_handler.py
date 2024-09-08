import pandas as pd
import pyarrow.parquet as pq
import json
import os
from typing import Dict, Tuple

class Raw_Data_Handler:
    def extract(self, customer_information_filename: str, transaction_filename: str, fraud_information_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Read customer data from CSV
        customer_information = pd.read_csv(customer_information_filename)
        
        # Read transaction data from Parquet
        transaction_information = pq.read_table(transaction_filename).to_pandas()
        transaction_information.reset_index(inplace=True, drop=False)

        # Read fraud data from JSON
        with open(fraud_information_filename, 'r') as file:
            fraud_data = json.load(file)
        
        # Convert fraud data to DataFrame
        fraud_information = pd.DataFrame.from_dict(fraud_data, orient='index').T
        fraud_information = fraud_information.melt(var_name='trans_num', value_name='is_fraud')
        fraud_information['trans_num'] = fraud_information['trans_num'].astype(str)
        
        return customer_information, transaction_information, fraud_information

    def transform(self, customer_information: pd.DataFrame, transaction_information: pd.DataFrame, fraud_information: pd.DataFrame) -> pd.DataFrame:
        # Merge transactions with customers
        merged_data = transaction_information.merge(customer_information, left_on='cc_num', right_on='cc_num', how='left')
        merged_data.set_index('trans_num', inplace=True)
        
        # Merge with fraud data
        merged_data = merged_data.merge(fraud_information, on='trans_num', how='left')

        # Standardize column names
        merged_data.columns = merged_data.columns.str.lower().str.replace(' ', '_')

        # Convert date columns to datetime
        merged_data['trans_date_trans_time'] = pd.to_datetime(merged_data['trans_date_trans_time'])
        merged_data['dob'] = pd.to_datetime(merged_data['dob'], format='mixed', dayfirst=False)
        
        # Create derived features
        merged_data['hour'] = merged_data['trans_date_trans_time'].dt.hour
        merged_data['day_of_week'] = merged_data['trans_date_trans_time'].dt.dayofweek
        merged_data['month'] = merged_data['trans_date_trans_time'].dt.month

        # Handle missing values
        merged_data['is_fraud'] = merged_data['is_fraud'].fillna(0)

        # Set index to trans_num and sort by trans_date_trans_time
        merged_data.set_index('trans_num', inplace=True)
        merged_data.sort_values('trans_date_trans_time', inplace=True)
    
        return merged_data

    def describe(self, raw_data: pd.DataFrame) -> Dict:
        description = {
            'version': 'v1.0',
            'storage': 'securebank/storage/raw_data/',
            'description': {
                'shape': raw_data.shape,
                'columns': raw_data.columns.tolist(),
                'dtypes': raw_data.dtypes.to_dict(),
                'missing_values': raw_data.isnull().sum().to_dict(),
                'fraud_ratio': raw_data['is_fraud'].mean()
            }
        }
        return description

    def load(self, raw_data: pd.DataFrame, output_filename: str) -> None:
        current_dir = os.getcwd()
        save_to_dir = os.path.join(os.path.dirname(current_dir), 'storage/raw_data')
        output_path = os.path.join(save_to_dir, output_filename)
        
        raw_data.to_parquet(output_path)