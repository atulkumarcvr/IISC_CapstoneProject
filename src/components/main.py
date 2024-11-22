from data_ingest import DataIngest
from data_transform import DataTransform
from model_train import ModelTrainer
import config
import sys
#from src.exception import CustomException
#from src

def main():
    try:
        # Step 1: Data ingestion
        print("Loading and splitting data...")
        ingest = DataIngest(config.DATA_DIR, config.LABEL_FILE, config.ARTIFACTS_DIR)
        data = ingest.load_data()
        X_train, X_test, y_train, y_test = ingest.split_data(data, test_size=config.TEST_SIZE)

        # Step 2: Data transformation
        print("Preprocessing data...")
        X_train, X_test = DataTransform.preprocess_data(X_train, X_test, config.ARTIFACTS_DIR)
        train_generator = DataTransform.augment_data(X_train, y_train)

        # Step 3: Model training
        print("Defining and training the model...")
        trainer = ModelTrainer(config.INPUT_SHAPE, config.NUM_CLASSES, config.ARTIFACTS_DIR)
        model = trainer.define_model()
        model, history = trainer.train_model(model, X_train, y_train, X_test, y_test, 
                                             epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        # Step 4: Model evaluation
        print("Evaluating the model...")
        loss, accuracy = trainer.evaluate_model(model, X_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    except Exception as e:
        #CustomException(e,sys)
         print(f"An error occurred in the pipeline: {e}")

if __name__ == "__main__":
    main()