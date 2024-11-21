from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle

class DataTransform:
    @staticmethod
    def preprocess_data(X_train, X_test, artifacts_dir):
        """
        Normalize image data and save as artifacts.
        """
        try:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
            
            # Save preprocessed data as artifacts
            with open(os.path.join(artifacts_dir, 'preprocessed_data.pkl'), 'wb') as f:
                pickle.dump((X_train, X_test), f)
            print("Preprocessed data saved as artifacts.")
            return X_train, X_test
        except Exception as e:
            print(f"Error in preprocessing data: {e}")
            raise

    @staticmethod
    def augment_data(X_train, y_train):
        """
        Apply data augmentation.
        """
        try:
            data_gen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1
            )
            return data_gen.flow(X_train, y_train, batch_size=32)
        except Exception as e:
            print(f"Error in augmenting data: {e}")
            raise