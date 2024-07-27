import os
import pandas as pd

# Create data directory if it doesn't exist
def create_data_directory(directory="data/extracted"):
    os.makedirs(directory, exist_ok=True)

# Function to save data to a file
def save_data(filename, data):
    data.to_csv(filename, index=False)

# Function to load data from a CSV file
def load_csv():
    file_path = "flat-files\Airline_Delay_Cause.csv"
    return pd.read_csv(file_path)

# Function to load all sheets from an XLSX file
def load_all_sheets():
    file_path = "flat-files\Airline_On_Time_Rankings.xlsx"
    try:
        excel_file = pd.ExcelFile(file_path)
        dfs = [pd.read_excel(file_path, sheet_name) for sheet_name in excel_file.sheet_names]
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function to extract and save data
def extract_and_save_data(data_dir="data/extracted"):
    create_data_directory(data_dir)
    
    # Load and save CSV data
    csv_data = load_csv()
    save_data(os.path.join(data_dir, "csv_data.csv"), csv_data)
    
    # Load and save XLSX data
    combined_xlsx_data = load_all_sheets()
    if not combined_xlsx_data.empty:
        save_data(os.path.join(data_dir, "xlsx_data_combined.csv"), combined_xlsx_data)
        print("XLSX data extracted and saved.")
    else:
        print("No data extracted from XLSX file due to an error.")

    print("Data extraction complete. Files saved in the 'data/' directory.")