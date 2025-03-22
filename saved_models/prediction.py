import os
import joblib

# Define all feature names based on the dataset
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Define the directory where models are stored
MODEL_DIR = r"D:\dataset\breast_cancer_detector\saved_models"  # Update with the correct path if necessary

# Load all models
try:
    print("Loading models...")
    gradient_boosting_model = joblib.load(os.path.join(MODEL_DIR, 'gradient_boosting_model.pkl'))
    random_forest_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit()

def get_best_model_prediction(input_features):
    """Get the best model's prediction and explanation in words, including low-confidence cases."""
    try:
        # Scale input features
        scaled_features = scaler.transform([input_features])

        # Predictions and probabilities
        gb_pred = gradient_boosting_model.predict(scaled_features)[0]
        gb_prob = gradient_boosting_model.predict_proba(scaled_features)[0][1]

        rf_pred = random_forest_model.predict(scaled_features)[0]
        rf_prob = random_forest_model.predict_proba(scaled_features)[0][1]

        # Determine the best model based on probabilities
        if gb_prob >= rf_prob:
            best_model = "Gradient Boosting"
            prediction = gb_pred
            probability = gb_prob
        else:
            best_model = "Random Forest"
            prediction = rf_pred
            probability = rf_prob

        # Translate prediction into words
        outcome = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-cancerous)"
        confidence = probability * 100  # Convert to percentage

        # Handle low-confidence cases
        if confidence < 75:
            return (f"The best model is {best_model}, but it is not confident in its prediction.\n"
                    f"The diagnosis is {outcome} with a low confidence level of {confidence:.2f}%. "
                    "Consider further medical evaluation.")
        else:
            return (f"The best model is {best_model}.\n"
                    f"The diagnosis is {outcome} with a confidence level of {confidence:.2f}%.")
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    # Predefined feature values (replace with your desired values)
    predefined_features = [
        12.34, 14.67, 78.94, 465.45, 0.0926, 0.0805, 0.0417, 0.0205, 0.1812, 0.0625,  
        0.295, 1.132, 2.284, 21.98, 0.0065, 0.0174, 0.0150, 0.0060, 0.0296, 0.0038,  
        14.56, 19.02, 95.33, 654.89, 0.1234, 0.1238, 0.0724, 0.0313,0.2426,0.0708
    ]

    # Ensure the predefined features match the number of feature names
    if len(predefined_features) != len(FEATURE_NAMES):
        print("Error: The number of predefined features does not match the number of feature names.")
        exit()

    # Display the predefined feature values
    print("Using predefined feature values:")
    for feature, value in zip(FEATURE_NAMES, predefined_features):
        print(f"{feature}: {value}")

    # Get the prediction
    result = get_best_model_prediction(predefined_features)
    print("\nPrediction Result:")
    print(result)