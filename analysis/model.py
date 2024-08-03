import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import pickle

def load_and_preprocess_data():
    """
    Load and preprocess flight and weather data from the specified CSV files.
    
    Returns:
        pd.DataFrame: Processed DataFrame with relevant features and a binary target variable.
    """
    flight_data_path = 'data/processed/csv_data_transformed.csv'
    weather_data_path = 'data/processed/weather_data.csv'

    # Load flight data with encoding handling
    try:
        flight_data = pd.read_csv(flight_data_path, encoding='utf-8')
    except UnicodeDecodeError:
        flight_data = pd.read_csv(flight_data_path, encoding='ISO-8859-1')
    
    # Ensure relevant columns are present in flight data
    if not {'arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct', 
            'arr_cancelled', 'arr_diverted', 'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 
            'security_delay', 'late_aircraft_delay', 'airport'}.issubset(flight_data.columns):
        raise ValueError("The flight data CSV file does not contain the required columns.")
    
    # Load weather data with encoding handling
    try:
        weather_data = pd.read_csv(weather_data_path, encoding='utf-8')
    except UnicodeDecodeError:
        weather_data = pd.read_csv(weather_data_path, encoding='ISO-8859-1')
    
    # Ensure relevant columns are present in weather data
    if not {'Location', 'Temperature (°C)', 'Conditions', 'Humidity (%)', 'Wind Speed (km/h)'}.issubset(weather_data.columns):
        raise ValueError("The weather data CSV file does not contain the required columns.")
    
    # Preprocess weather data
    weather_data.rename(columns={'Location': 'airport', 'Temperature (°C)': 'temperature', 
                                 'Humidity (%)': 'humidity', 'Wind Speed (km/h)': 'wind_speed'}, inplace=True)
    
    # Merge flight data with weather data based on the airport location
    merged_data = pd.merge(flight_data, weather_data, on='airport', how='left')
    
    # Fill missing weather data with default values
    merged_data.fillna({'temperature': 0, 'humidity': 0, 'wind_speed': 0}, inplace=True)
    
    # Create a binary target variable for demonstration
    merged_data['cancellation'] = (merged_data['arr_cancelled'] > 0).astype(int)
    
    return merged_data

def train_model(data):
    """
    Train and evaluate a logistic regression model on the given data.

    Args:
        data (pd.DataFrame): The processed DataFrame with features and target variable.

    Returns:
        sklearn.pipeline.Pipeline: Trained logistic regression model.
    """
    features = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct',
                'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay',
                'temperature', 'humidity', 'wind_speed']
    
    X = data[features]
    y = data['cancellation']
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    return model

def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model (sklearn.pipeline.Pipeline): The trained model to be saved.
        file_path (str): Path to the file where the model will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)