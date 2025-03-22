import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.breast_cancer_detector import BreastCancerDetector

def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'data.csv')
    models_path = os.path.join(current_dir, '..', 'saved_models')
    
    # Initialize detector
    detector = BreastCancerDetector(data_path)
    
    # Load and preprocess data
    if detector.load_and_prepare_data():
        detector.preprocess_data()
        
        # Train models
        detector.train_models()
        
        # Analyze features
        detector.analyze_feature_importance()
        
        # Evaluate best model
        detector.evaluate_model()
        
        # Make prediction with example data
        example_features = np.random.rand(len(detector.X.columns))
        result = detector.predict_cancer(example_features)
        print("\nPrediction Results:")
        print(f"Diagnosis: {result['prediction']}")
        print(f"Probability: {result['probability']:.2f}")
        print(f"95% Confidence Interval: ({result['confidence_interval'][0]:.2f}, "
              f"{result['confidence_interval'][1]:.2f})")
        
        # Save models
        detector.save_models(models_path)

if __name__ == "__main__":
    main()