from etl import extract, load, transform
from analysis import model as modelTrain, evaluate

def main():
    # Instantiate objects
    extractionObject = extract
    loadObject = load
    transformObject = transform
    modelObject = modelTrain
    evaluateObject = evaluate

    # Step 1: Data Extraction
    try:
        # Extract and save data
        extractionObject.extract_and_save_data()
        weatherJson = extractionObject.get_weather()
        if not weatherJson:
            raise ValueError("Weather JSON data is empty or invalid.")
        print("Data extraction completed.")
    except ValueError as ve:
        print(f"Data extraction error: {ve}")
    except Exception as e:
        print(f"Data extraction failed: {str(e)}")

    # Step 2: Data Transformation
    try:
        # Transform and save data
        transformObject.transform_data()
        transformObject.save_weather_to_csv(weatherJson)
        print("Data transformation completed.")
    except FileNotFoundError as fnf_error:
        print(f"Data transformation failed: File not found - {fnf_error}")
    except PermissionError as perm_error:
        print(f"Data transformation failed: Permission denied - {perm_error}")
    except Exception as e:
        print(f"Data transformation failed: {str(e)}")

    # Step 3: Data Loading
    try:
        # Load and analyze data
        loadObject.load_and_analyze_data()
        print("Data loading completed.")
    except FileNotFoundError as fnf_error:
        print(f"Data loading failed: File not found - {fnf_error}")
    except ValueError as ve:
        print(f"Data loading failed: Value error - {ve}")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")

    # Step 4: Model Training
    try:
        # Load data, train model, and save the model
        data = modelObject.load_and_preprocess_data()
        model = modelObject.train_model(data)
        modelObject.save_model(model, 'data/outputs/model.pkl')
        print("Model training completed.")
    except FileNotFoundError as fnf_error:
        print(f"Model training failed: File not found - {fnf_error}")
    except ValueError as ve:
        print(f"Model training failed: Value error - {ve}")
    except PermissionError as perm_error:
        print(f"Model training failed: Permission denied - {perm_error}")
    except Exception as e:
        print(f"Data model training failed: {str(e)}")

    # Step 5: Model Evaluation
    try:
        evaluateObject.evaluate_model()
        print("Model evaluation completed.")
    except FileNotFoundError as fnf_error:
        print(f"Model evaluation failed: File not found - {fnf_error}")
    except ValueError as ve:
        print(f"Model evaluation failed: Value error - {ve}")
    except PermissionError as perm_error:
        print(f"Model evaluation failed: Permission denied - {perm_error}")
    except Exception as e:
        print(f"Data evaluation failed: {str(e)}")

    # Step 6: Visualization
    try:
        print("Starting data visualization...")
        # Code to create visualizations
        print("Data visualization completed.")
    except Exception as e:
        print(f"Data visualization failed: {str(e)}")

    # Output placeholder for evaluation metrics
    print("Evaluation Metrics: Placeholder")

if __name__ == "__main__":
    main()
