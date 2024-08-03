# main.py

from etl import extract, load, transform
from analysis import model as modelTrain, evaluate

def main():
    #Instantiate objects
    extractionObject = extract
    loadObject = load
    transformObject = transform
    modelObject = modelTrain
    evaluateObject = evaluate
    # Basic setup for the project workflow

    # Step 1: Data Extraction
    try:
        # Extract and save data
        extractionObject.extract_and_save_data()
        weatherJson = extractionObject.get_weather()
        print("Data extraction completed.")
    except Exception as e:
        print(f"Data extraction failed: {str(e)}")

    
    # Step 2: Data Transformation
    try:
        # Extract and save data
        transformObject.transform_data()
        transformObject.save_weather_to_csv(weatherJson)
        print("Data transformation completed.")
    except Exception as e:
        print(f"Data transformation failed: {str(e)}")
    
    # Step 3: Data Loading
    try:
        # Extract and save data
        loadObject.load_and_analyze_data()
        print("Data loading completed.")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
    
    # Step 4: Model Training
    try:
          data = modelObject.load_and_preprocess_data()
          model = modelObject.train_model(data)
          modelObject.save_model(model, 'data/outputs/model.pkl')
    except Exception as e:
        print(f"Data model training failed: {str(e)}")
    
    # Step 5: Model Evaluation
    try:
        evaluateObject.evaluate_model()
    except Exception as e:
        print(f"Data evaluation failed: {str(e)}")
    # Step 6: Visualization
    print("Starting data visualization...")
    # Code to create visualizations
    print("Data visualization completed.")
    
    # Output placeholder for evaluation metrics
    print("Evaluation Metrics: Placeholder")

if __name__ == "__main__":
    main()