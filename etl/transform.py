import os
import pandas as pd

# Create processed data directory if it doesn't exist
def create_processed_data_directory(directory="data/processed"):
    os.makedirs(directory, exist_ok=True)

# Load extracted data
def load_extracted_data():
    csv_data_path = "data/extracted/csv_data.csv"
    xlsx_data_path = "data/extracted/xlsx_data_combined.csv"
    
    # Ensure the files exist
    if not os.path.exists(csv_data_path):
        raise FileNotFoundError(f"CSV file not found at {csv_data_path}")
    if not os.path.exists(xlsx_data_path):
        raise FileNotFoundError(f"XLSX file not found at {xlsx_data_path}")
    
    csv_data = pd.read_csv(csv_data_path)
    xlsx_data = pd.read_csv(xlsx_data_path)
    
    return csv_data, xlsx_data

# Clean and wrangle data
def clean_and_wrangle_data(csv_data, xlsx_data):
    try:
        # Remove rows where any cell contains 'Unknown' in both datasets
        csv_data = csv_data[~csv_data.apply(lambda row: row.str.contains('Unknown').any(), axis=1)]
        xlsx_data = xlsx_data[~xlsx_data.apply(lambda row: row.str.contains('Unknown').any(), axis=1)]
        
        # Drop any rows that are entirely empty in xlsx_data
        xlsx_data = xlsx_data.dropna(how='all')
        
        # Remove unnecessary rows (header rows) and reset index
        xlsx_data = xlsx_data.reset_index(drop=True)
        
        # Assuming the actual header is in row 1 (index 1)
        xlsx_data.columns = xlsx_data.iloc[0]  # Set header from the first row
        xlsx_data = xlsx_data[1:]  # Remove the header row
        xlsx_data.reset_index(drop=True, inplace=True)
        
        # Remove any columns that are completely empty or contain 'Unknown'
        xlsx_data = xlsx_data.loc[:, (xlsx_data != 'Unknown').any(axis=0)]
        
        # Remove any remaining completely empty rows
        xlsx_data = xlsx_data.dropna(how='all')
        
        # Clean up csv_data by dropping empty columns
        csv_data = csv_data.dropna(axis=1, how='all')
        
        # Drop duplicates
        csv_data = csv_data.drop_duplicates()
        xlsx_data = xlsx_data.drop_duplicates()
        
        # Consistent attribute naming for both csv_data and xlsx_data
        csv_data.columns = csv_data.columns.str.lower().str.replace(' ', '_')
        xlsx_data.columns = xlsx_data.columns.str.lower().str.replace(' ', '_')
        
        # Optional: Remove any remaining completely empty columns
        csv_data = csv_data.dropna(axis=1, how='all')
        
        return csv_data, xlsx_data
    except Exception as e:
        print(f"Error during data transformation: {str(e)}")


# Save transformed data
def save_transformed_data(csv_data, xlsx_data, directory="data/processed"):
    # Ensure the directory exists
    create_processed_data_directory(directory)
    
    # Save each dataset to a separate CSV file
    csv_file_path = os.path.join(directory, "csv_data_transformed.csv")
    xlsx_file_path = os.path.join(directory, "xlsx_data_transformed.csv")
    
    csv_data.to_csv(csv_file_path, index=False)
    xlsx_data.to_csv(xlsx_file_path, index=False)
    
    print(f"Transformed CSV data saved to {csv_file_path}")
    print(f"Transformed XLSX data saved to {xlsx_file_path}")

# Main function to orchestrate transformation
def transform_data():
    create_processed_data_directory()
    
    csv_data, xlsx_data = load_extracted_data()
    csv_data_cleaned, xlsx_data_cleaned = clean_and_wrangle_data(csv_data, xlsx_data)
    
    save_transformed_data(csv_data_cleaned, xlsx_data_cleaned)