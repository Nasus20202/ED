import pandas as pd
import json
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import centralized configuration
from config import *

def load_and_prepare_data():
    """Load and prepare flight data using configuration parameters"""
    main_data = pd.read_csv(MAIN_DATA_FILE_NAME)
    code_to_airport = pd.read_csv(CODE_TO_AIRPORT_FILE_NAME)
    code_to_airline = pd.read_csv(CODE_TO_AIRLINE_FILE_NAME)

    # Create delayed target variable
    main_data[TARGET] = (main_data["ARRIVAL_DELAY"] > DELAYED_THRESHOLD).astype(int)

    # Fill missing values and filter
    main_data = main_data.fillna(FILL_NA_VALUE)
    main_data = main_data.dropna(subset=[TARGET])

    # Balance the dataset
    delayed_flights = main_data[main_data[TARGET] == 1]
    non_delayed_flights = main_data[main_data[TARGET] == 0]

    print(f"Original delayed flights: {len(delayed_flights)}")
    print(f"Original non-delayed flights: {len(non_delayed_flights)}")

    # Take minimum of both classes to balance
    data_size_limit = get_data_size_limit("model_comparison")
    min_samples = min(min(len(delayed_flights), len(non_delayed_flights)), data_size_limit)
    balanced_delayed = delayed_flights.sample(n=min_samples, random_state=RANDOM_STATE)
    balanced_non_delayed = non_delayed_flights.sample(n=min_samples, random_state=RANDOM_STATE)

    # Combine balanced data
    main_data = pd.concat([balanced_delayed, balanced_non_delayed]).reset_index(drop=True)

    print(f"Balanced delayed flights: {(main_data[TARGET] == 1).sum()}")
    print(f"Balanced non-delayed flights: {(main_data[TARGET] == 0).sum()}")

    # Encode categorical variables
    X = main_data[CATEGORIES + NUMERICAL].copy()
    for col in CATEGORIES:
        X[col] = X[col].astype("category").cat.codes

    y = main_data[TARGET]
    return X, y

def create_models():
    """Create model instances using configuration parameters"""
    models = {}
    
    for model_name in MODELS_TO_COMPARE:
        params = DEFAULT_MODEL_PARAMS[model_name].copy()
        
        if model_name == "RandomForest":
            models[model_name] = RandomForestClassifier(**params)
        elif model_name == "LogisticRegression":
            models[model_name] = LogisticRegression(**params)
        elif model_name == "DecisionTree":
            models[model_name] = DecisionTreeClassifier(**params)
        elif model_name == "KNN":
            models[model_name] = KNeighborsClassifier(**params)
        elif model_name == "NeuralNetwork":
            models[model_name] = MLPClassifier(**params)
    
    return models

def main():
    """Main comparison function"""
    # Validate configuration
    validate_config()
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Create models
    models = create_models()
    
    results = {}
    detailed_results = {}

    for name, model in models.items():
        if REPORTING["progress_bar"]:
            tqdm.write(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        # Store detailed results
        detailed_results[name] = {
            "accuracy": float(acc),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        if REPORTING["verbose"]:
            tqdm.write(f"{name} Results:")
            tqdm.write(f"Accuracy: {acc:.{REPORTING['precision_digits']}f}")
            tqdm.write(classification_report(y_test, y_pred, digits=REPORTING['classification_report_digits']))

    # Print summary table
    if REPORTING["verbose"]:
        tqdm.write("\nSummary of Model Accuracies:")
        for name, acc in results.items():
            tqdm.write(f"{name:15}: {acc:.{REPORTING['precision_digits']}f}")

    # Save results to file
    save_data = {
        "dataset_info": {
            "shape": X.shape,
            "target_distribution": y.value_counts().to_dict(),
            "test_size": TEST_SIZE,
            "delayed_threshold": DELAYED_THRESHOLD
        },
        "model_results": detailed_results,
        "summary": results,
        "configuration": {
            "models_compared": MODELS_TO_COMPARE,
            "data_size_limit": get_data_size_limit("model_comparison"),
            "random_state": RANDOM_STATE
        }
    }

    with open(OUTPUT_FILES["model_comparison"], 'w') as f:
        json.dump(save_data, f, indent=2)

    tqdm.write(f"\nResults saved to {OUTPUT_FILES['model_comparison']}")

if __name__ == "__main__":
    main()
