import os
import sys
import time
import pandas as pd
import numpy as np
from config.model_config import MODEL_CONFIGS

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

class BreastCancerDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.best_model = None

    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        try:
            print("Loading data...")
            time.sleep(1)
            self.data = pd.read_csv(self.data_path)
            print("Data loaded successfully.")

            # Drop unnecessary columns
            print("Preparing data...")
            self.data.drop(columns=['id'], inplace=True)

            # Map target variable to binary
            self.data['diagnosis'] = self.data['diagnosis'].map({'B': 0, 'M': 1})
            print("Data preparation complete.")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def preprocess_data(self, test_size=0.2):
        """Preprocess data with scaling (no feature selection)"""
        try:
            print("Preprocessing data...")
            time.sleep(1)
            self.X = self.data.drop(columns=['diagnosis'])  # Use all 30 features
            self.y = self.data['diagnosis']

            # Train-test split
            print("Splitting data into training and testing sets...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
            )

            # Scale features
            print("Scaling data...")
            self.scaler = RobustScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

            print("Data preprocessing completed.")
            return True
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return False

    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        for name, config in MODEL_CONFIGS.items():
            print(f"\nTraining {name}...")
            time.sleep(1)
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=StratifiedKFold(n_splits=5),
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            self.models[name] = grid_search.best_estimator_

            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score for {name}: {grid_search.best_score_:.3f}")

        self.best_model = max(self.models.items(), key=lambda x: x[1].score(self.X_test, self.y_test))
        print(f"\nBest model: {self.best_model[0]} with score: {self.best_model[1].score(self.X_test, self.y_test):.3f}")

    def evaluate_model(self, model_name=None):
        """Evaluate model with metrics and visualizations"""
        print("\nEvaluating model...")
        time.sleep(1)
        model = self.models[model_name] if model_name else self.best_model[1]

        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        print("\nModel Evaluation Metrics:")
        print(classification_report(self.y_test, y_pred))

        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def save_models(self, path):
        """Save trained models and scaler"""
        try:
            print("Saving models...")
            time.sleep(1)
            os.makedirs(path, exist_ok=True)
            for name, model in self.models.items():
                joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))
            joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
            print(f"Models saved to {path}")
        except Exception as e:
            print(f"Error saving models: {str(e)}")

if __name__ == "__main__":
    print("Welcome to the Breast Cancer Detector!")
    data_path = input("Enter the path to the breast cancer dataset (data.csv): ")
    detector = BreastCancerDetector(data_path)

    if detector.load_and_prepare_data():
        if detector.preprocess_data():
            detector.train_models()
            detector.evaluate_model()

            save_path = input("Enter the directory to save the models: ")
            detector.save_models(save_path)

            print("\nAll tasks completed successfully!")
        else:
            print("Data preprocessing failed.")
    else:
        print("Data loading failed.")