import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import MODEL_CONFIGS, DATA_CONFIG

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
        self.feature_importance = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset with enhanced error handling and data validation"""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File not found: {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            
            # Data validation
            required_columns = ['diagnosis', 'id']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError("Required columns missing from dataset")
            
            # Convert diagnosis and drop ID
            self.data['diagnosis'] = self.data['diagnosis'].map({'B': 0, 'M': 1})
            self.data = self.data.drop(columns=['id'])
            
            # Handle outliers using IQR method
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | 
                                  (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]
            
            return True
            
        except Exception as e:
            print(f"Error in data loading: {str(e)}")
            return False
    
    def preprocess_data(self, test_size=0.2):
        """Enhanced preprocessing with multiple scaling options and feature selection"""
        try:
            # Separate features and target
            self.X = self.data.drop(columns=['diagnosis'])
            self.y = self.data['diagnosis']
            
            # Feature selection using Random Forest
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
            selector.fit(self.X, self.y)
            selected_features = self.X.columns[selector.get_support()].tolist()
            self.X = self.X[selected_features]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
            )
            
            # Scale features
            self.scaler = RobustScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            return True
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return False
    
    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        for name, config in MODEL_CONFIGS.items():
            print(f"\nTraining {name}...")
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=StratifiedKFold(n_splits=5),
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            self.models[name] = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Select best model
        best_score = 0
        for name, model in self.models.items():
            score = model.score(self.X_test, self.y_test)
            if score > best_score:
                best_score = score
                self.best_model = (name, model)
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        if 'random_forest' in self.models:
            model = self.models['random_forest']
            importance = model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=self.feature_importance.head(10))
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            plt.show()
    
    def evaluate_model(self, model_name=None):
        """Comprehensive model evaluation with visualizations"""
        model = self.models[model_name] if model_name else self.best_model[1]
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:,1]
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        print(classification_report(self.y_test, y_pred))
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def predict_cancer(self, features):
        """Make predictions with confidence intervals"""
        if len(features) != len(self.X.columns):
            raise ValueError("Number of features doesn't match the model requirements")
        
        features_scaled = self.scaler.transform([features])
        probabilities = {}
        
        for name, model in self.models.items():
            prob = model.predict_proba(features_scaled)[0][1]
            probabilities[name] = prob
        
        # Calculate ensemble prediction and confidence interval
        probs = list(probabilities.values())
        mean_prob = np.mean(probs)
        ci = stats.t.interval(0.95, len(probs)-1, loc=mean_prob, 
                            scale=stats.sem(probs))
        
        return {
            'prediction': 'Malignant' if mean_prob > 0.5 else 'Benign',
            'probability': mean_prob,
            'confidence_interval': ci,
            'individual_model_predictions': probabilities
        }
    
    def save_models(self, path):
        """Save trained models and scalers"""
        try:
            os.makedirs(path, exist_ok=True)
            for name, model in self.models.items():
                joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))
            joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
            print(f"Models saved successfully to {path}")
        except Exception as e:
            print(f"Error saving models: {str(e)}")