from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import pickle

class ModelTrainer:
    def __init__(self, input_shape, num_classes, artifacts_dir):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def define_model(self):
        """
        Define the CNN model.
        """
        try:
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                MaxPooling2D((2, 2)),
                Dropout(0.2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            print(f"Error in defining the model: {e}")
            raise

    def train_model(self, model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        """
        Train the CNN model and save the model and training history.
        """
        try:
            y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
            y_test_cat = to_categorical(y_test, num_classes=self.num_classes)

            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Save the trained model and history
            model_path = os.path.join(self.artifacts_dir, 'trained_model.h5')
            model.save(model_path)
            with open(os.path.join(self.artifacts_dir, 'training_history.pkl'), 'wb') as f:
                pickle.dump(history.history, f)
            print(f"Model and training history saved to {self.artifacts_dir}.")
            return model, history
        except Exception as e:
            print(f"Error in training the model: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model.
        """
        try:
            y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
            loss, accuracy = model.evaluate(X_test, y_test_cat)
            return loss, accuracy
        except Exception as e:
            print(f"Error in evaluating the model: {e}")
            raise
