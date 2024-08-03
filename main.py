import logging
from etl import extract, load, transform
from analysis import model as modelTrain, evaluate

# Configure logging
logging.basicConfig(filename='data_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info('Started data pipeline execution')

    # Step 1: Data Extraction
    try:
        logger.info('Starting data extraction')
        extractionObject = extract
        extractionObject.extract_and_save_data()
        weatherJson = extractionObject.get_weather()
        if not weatherJson:
            raise ValueError("Weather JSON data is empty or invalid.")
        logger.info('Data extraction completed')
    except ValueError as ve:
        logger.error(f"Data extraction error: {ve}")
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")

    # Step 2: Data Transformation
    try:
        logger.info('Starting data transformation')
        transformObject = transform
        transformObject.transform_data()
        transformObject.save_weather_to_csv(weatherJson)
        logger.info('Data transformation completed')
    except FileNotFoundError as fnf_error:
        logger.error(f"Data transformation failed: File not found - {fnf_error}")
    except PermissionError as perm_error:
        logger.error(f"Data transformation failed: Permission denied - {perm_error}")
    except Exception as e:
        logger.error(f"Data transformation failed: {str(e)}")

    # Step 3: Data Loading
    try:
        logger.info('Starting data loading')
        loadObject = load
        loadObject.load_and_analyze_data()
        logger.info('Data loading completed')
    except FileNotFoundError as fnf_error:
        logger.error(f"Data loading failed: File not found - {fnf_error}")
    except ValueError as ve:
        logger.error(f"Data loading failed: Value error - {ve}")
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")

    # Step 4: Model Training
    try:
        logger.info('Starting model training')
        modelObject = modelTrain
        data = modelObject.load_and_preprocess_data()
        model = modelObject.train_model(data)
        modelObject.save_model(model, 'data/outputs/model.pkl')
        logger.info('Model training completed')
    except FileNotFoundError as fnf_error:
        logger.error(f"Model training failed: File not found - {fnf_error}")
    except ValueError as ve:
        logger.error(f"Model training failed: Value error - {ve}")
    except PermissionError as perm_error:
        logger.error(f"Model training failed: Permission denied - {perm_error}")
    except Exception as e:
        logger.error(f"Data model training failed: {str(e)}")

    # Step 5: Model Evaluation
    try:
        logger.info('Starting model evaluation')
        evaluateObject = evaluate
        evaluateObject.evaluate_model()
        logger.info('Model evaluation completed')
    except FileNotFoundError as fnf_error:
        logger.error(f"Model evaluation failed: File not found - {fnf_error}")
    except ValueError as ve:
        logger.error(f"Model evaluation failed: Value error - {ve}")
    except PermissionError as perm_error:
        logger.error(f"Model evaluation failed: Permission denied - {perm_error}")
    except Exception as e:
        logger.error(f"Data evaluation failed: {str(e)}")

    # Step 6: Visualization
    try:
        logger.info('Starting data visualization')
        # Code to create visualizations
        logger.info('Data visualization completed')
    except Exception as e:
        logger.error(f"Data visualization failed: {str(e)}")

    logger.info('Data pipeline execution finished')

if __name__ == "__main__":
    main()
