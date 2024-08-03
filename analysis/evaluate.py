import pickle
import pandas as pd
from analysis import model as modelTrain
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_model():
    """
    Load and evaluate the model using the provided data.
    """
    model_path = 'data/outputs/model.pkl'

    # Load and preprocess data
    data = modelTrain.load_and_preprocess_data()
    
    features = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct',
                'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay',
                'temperature', 'humidity', 'wind_speed']
    
    X = data[features]
    y = data['cancellation']
    
    # Drop rows with missing values
    X.dropna(inplace=True)
    y = y.loc[X.index]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
