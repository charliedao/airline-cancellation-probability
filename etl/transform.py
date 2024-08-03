import csv
import os
import pandas as pd

# Create processed data directory if it doesn't exist
def create_processed_data_directory(directory="data/processed"):
    """
    Create a directory for storing processed data if it doesn't already exist.

    Args:
        directory (str): The path of the directory to create. Default is "data/processed".
    """
    os.makedirs(directory, exist_ok=True)

# Load extracted data
def load_extracted_data():
    """
    Load extracted data from CSV and XLSX files.

    Returns:
        tuple: A tuple containing two DataFrames: the CSV data and the XLSX data.

    Raises:
        FileNotFoundError: If the CSV or XLSX file is not found at the specified paths.
    """
    csv_data_path = "data/outputs/csv_data.csv"
    xlsx_data_path = "data/outputs/xlsx_data_combined.csv"
    
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
    """
    Clean and wrangle the given CSV and XLSX data.

    Args:
        csv_data (pd.DataFrame): The DataFrame containing CSV data.
        xlsx_data (pd.DataFrame): The DataFrame containing XLSX data.

    Returns:
        tuple: A tuple containing two cleaned DataFrames: the cleaned CSV data and the cleaned XLSX data.
    
    Prints:
        - Error message if any exception occurs during the cleaning and wrangling process.
    """
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
    """
    Save the cleaned and wrangled data to CSV files in the specified directory.

    Args:
        csv_data (pd.DataFrame): The cleaned CSV data to save.
        xlsx_data (pd.DataFrame): The cleaned XLSX data to save.
        directory (str): The path of the directory where the transformed data will be saved. Default is "data/processed".
    
    Prints:
        - Confirmation messages for saved files.
    """
    # Ensure the directory exists
    create_processed_data_directory(directory)
    
    # Save each dataset to a separate CSV file
    csv_file_path = os.path.join(directory, "csv_data_transformed.csv")
    xlsx_file_path = os.path.join(directory, "xlsx_data_transformed.csv")
    
    csv_data.to_csv(csv_file_path, index=False)
    xlsx_data.to_csv(xlsx_file_path, index=False)
    
    print(f"Transformed CSV data saved to {csv_file_path}")
    print(f"Transformed XLSX data saved to {xlsx_file_path}")

# Perform Exploratory Data Analysis (EDA)
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
    print(data.describe(include='all'), "\n")
    
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum(), "\n")

# Main function to orchestrate transformation
def transform_data():
    """
    Orchestrate the transformation process by loading, cleaning, and saving the data.
    
    This function:
    - Creates the processed data directory.
    - Loads the extracted data.
    - Cleans and wrangles the data.
    - Saves the transformed data to CSV files.
    - Performs EDA on the cleaned data.
    """
    create_processed_data_directory()
    
    csv_data, xlsx_data = load_extracted_data()
    csv_data_cleaned, xlsx_data_cleaned = clean_and_wrangle_data(csv_data, xlsx_data)
    
    save_transformed_data(csv_data_cleaned, xlsx_data_cleaned)
    
    # Perform EDA on cleaned data
    perform_eda(csv_data_cleaned, "CSV Data")
    perform_eda(xlsx_data_cleaned, "XLSX Data")

def save_weather_to_csv(data):
    """
    Save the weather data to a CSV file in the 'data/processed' directory.

    Args:
        data (dict): The weather data to save. This dictionary should contain current weather conditions including location,
                     temperature, conditions, humidity, and wind speed.
    
    The function creates the 'data/processed' directory if it doesn't exist and appends weather data to a CSV file. 
    It writes a header only if the file does not already exist.
    """
    directory = "data/processed"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, "weather_data.csv")
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Write the header
            writer.writerow(["Location", "Temperature (°C)", "Conditions", "Humidity (%)", "Wind Speed (km/h)"])
        
        current_conditions = data.get('currentConditions', {})
        writer.writerow([
            data.get('resolvedAddress', 'N/A'),
            current_conditions.get('temp', 'N/A'),
            current_conditions.get('conditions', 'N/A'),
            current_conditions.get('humidity', 'N/A'),
            current_conditions.get('windspeed', 'N/A')
        ])