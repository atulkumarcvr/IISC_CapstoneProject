import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pickle

class DataIngest:
    def __init__(self, data_dir, label_file, artifacts_dir):
        self.data_dir = data_dir
        self.label_file = label_file
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def load_data(self):
        """
        Load images and labels from the dataset.
        """
        try:
            labels = pd.read_csv(self.label_file)
            data = []
            for _, row in labels.iterrows():
                img_path = os.path.join(self.data_dir, row['Path'])
                image = cv2.imread(img_path)
                image = cv2.resize(image, (50, 50)) # Resize images to 50x50
                data.append((image, row['ClassId']))
            return data
        except Exception as e:
            print(f"Error in loading data: {e}")
            raise

    def split_data(self, data, test_size=0.2):
        """
        Split data into training and testing sets.
        """
        try:
            X = np.array([item[0] for item in data])
            y = np.array([item[1] for item in data])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Save split data as artifacts
            with open(os.path.join(self.artifacts_dir, 'train_test_split.pkl'), 'wb') as f:
                pickle.dump((X_train, X_test, y_train, y_test), f)
            print("Data split and saved as artifacts.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error in splitting data: {e}")
            raise
