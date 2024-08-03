import os
import pandas as pd
import requests

# Create data directory if it doesn't exist
def create_data_directory(directory="data/extracted"):
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): The path of the directory to create. Default is "data/extracted".
    """
    os.makedirs(directory, exist_ok=True)

# Function to save data to a file
def save_data(filename, data):
    """
    Save data to a CSV file.

    Args:
        filename (str): The name of the file to save the data.
        data (pd.DataFrame): The data to save.
    """
    data.to_csv(filename, index=False)

# Function to load data from a CSV file
def load_csv():
    """
    Load data from a CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    file_path = "data\extracted\Airline_Delay_Cause.csv"
    return pd.read_csv(file_path)

# Function to load all sheets from an XLSX file
def load_all_sheets():
    """
    Load all sheets from an XLSX file and combine them into a single DataFrame.

    Returns:
        pd.DataFrame: The combined data from all sheets.
        If there is an error, returns an empty DataFrame.
    """
    file_path = "data\extracted\Airline_On_Time_Rankings.xlsx"
    try:
        excel_file = pd.ExcelFile(file_path)
        dfs = [pd.read_excel(file_path, sheet_name) for sheet_name in excel_file.sheet_names]
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function to extract and save data
def extract_and_save_data(data_dir="data/outputs"):
    """
    Extract data from CSV and XLSX files and save them to the specified directory.

    Args:
        data_dir (str): The directory to save the extracted data. Default is "data/outputs".
    """
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

def get_weather():
    """
    Get the current weather for Maryland using the Visual Crossing Weather API.

    Returns:
        dict: A dictionary containing the current weather data.
    """
    api_key = "QUPRUBJ6UR9HWEJZT653QRJBD"
    location = "Maryland"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}?unitGroup=metric&key={api_key}&contentType=json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to get weather data. HTTP Status code: {response.status_code}"}
