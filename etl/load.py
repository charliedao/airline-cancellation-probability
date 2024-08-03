import os
import pandas as pd

# Function to load the processed data
def load_processed_data(directory="data/processed"):
    """
    Load the transformed data from CSV and XLSX files in the specified directory.

    Args:
        directory (str): The path of the directory where the transformed data is stored. Default is "data/processed".

    Returns:
        tuple: A tuple containing two DataFrames: the CSV data and the XLSX data.

    Raises:
        FileNotFoundError: If the transformed CSV or XLSX file is not found at the specified paths.
    """
    csv_file_path = os.path.join(directory, "csv_data_transformed.csv")
    xlsx_file_path = os.path.join(directory, "xlsx_data_transformed.csv")
    
    # Ensure the files exist
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Transformed CSV file not found at {csv_file_path}")
    if not os.path.exists(xlsx_file_path):
        raise FileNotFoundError(f"Transformed XLSX file not found at {xlsx_file_path}")
    
    csv_data = pd.read_csv(csv_file_path)
    xlsx_data = pd.read_csv(xlsx_file_path)
    
    return csv_data, xlsx_data

# Function to perform EDA without visualizations
def perform_eda(data, title="Dataset"):
    """
    Perform exploratory data analysis (EDA) on the given dataset.

    Args:
        data (pd.DataFrame): The dataset on which to perform EDA.
        title (str): The title for the EDA output. Default is "Dataset".

    Prints:
        - The first few rows of the dataset.
        - Summary statistics of the dataset.
        - Missing values in the dataset.
    """
    print(f"\nEDA for {title}:\n")
    
    # Print the first few rows of the dataset
    print("First few rows:")
    print(data.head(), "\n")
    
    # Print summary statistics
    print("Summary statistics:")
    print(data.describe(), "\n")
    
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum(), "\n")

# Main function to orchestrate loading and EDA
def load_and_analyze_data():
    """
    Load the transformed data and perform exploratory data analysis (EDA) on both CSV and XLSX datasets.
    
    This function loads the processed data from CSV and XLSX files, prints the shapes of the datasets,
    and then performs EDA on both datasets.
    """
    csv_data, xlsx_data = load_processed_data()
    
    # Print out the shape of the loaded data for verification
    print(f"CSV Data Shape: {csv_data.shape}")
    print(f"XLSX Data Shape: {xlsx_data.shape}")
    
    # Perform EDA on both datasets
    perform_eda(csv_data, title="CSV Data")
    perform_eda(xlsx_data, title="XLSX Data")
