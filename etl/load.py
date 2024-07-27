import os
import pandas as pd

# Load data from CSV files
def load_data_from_csvs(directory="data"):
    csv_data_path = os.path.join(directory, "csv_data.csv")
    xlsx_data_path = os.path.join(directory, "transformed_data.csv")
    
    csv_data = pd.read_csv(csv_data_path)
    xlsx_data = pd.read_csv(xlsx_data_path)
    
    return csv_data, xlsx_data

# Perform Exploratory Data Analysis (EDA)
def perform_eda(data, name):
    print(f"\nEDA for {name}:")
    print("Data Info:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nDuplicates:")
    print(data.duplicated().sum())
    print("\nFirst 5 Rows of Data:")
    print(data.head())
