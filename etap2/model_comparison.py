import pandas as pd
import json
from tqdm import tqdm
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn import tree

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
    # Ensure img directory exists
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './img'))
    os.makedirs(img_dir, exist_ok=True)

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

    # --- Visualization Section ---
    # 1. Bar chart of model accuracies
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracies Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "model_accuracies.png"))
    plt.close()

    # 2. Confusion matrices for each model
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(os.path.join(img_dir, f"confusion_matrix_{name}.png"))
        plt.close()

    # 3. Visualize Decision Tree (if present)
    if "DecisionTree" in models:
        fig = plt.figure(figsize=(16, 8))
        tree.plot_tree(models["DecisionTree"], feature_names=X.columns, class_names=["On Time", "Delayed"], filled=True, max_depth=2)
        plt.title("Decision Tree Visualization (max_depth=2)")
        plt.savefig(os.path.join(img_dir, "decision_tree_visualization.png"))
        plt.close()

    # 3b. Visualize a Random Forest tree (first estimator)
    if "RandomForest" in models:
        rf_model = models["RandomForest"]
        if hasattr(rf_model, "estimators_") and len(rf_model.estimators_) > 0:
            fig = plt.figure(figsize=(16, 8))
            tree.plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=["On Time", "Delayed"], filled=True, max_depth=2)
            plt.title("Random Forest: First Tree Visualization (max_depth=2)")
            plt.savefig(os.path.join(img_dir, "random_forest_tree_visualization.png"))
            plt.close()

    # 4. KNN centers (PCA-reduced scatter plot)
    if "KNN" in models:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap="coolwarm", alpha=0.5, label="Samples")
        plt.title("KNN: PCA Projection of Test Data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(scatter, label="Class")
        plt.savefig(os.path.join(img_dir, "knn_pca_scatter.png"))
        plt.close()

    # 5. Neural Network loss curve (if available)
    if "NeuralNetwork" in models:
        nn_model = models["NeuralNetwork"]
        if hasattr(nn_model, "loss_curve_"):
            plt.figure(figsize=(7, 4))
            plt.plot(nn_model.loss_curve_)
            plt.title("Neural Network Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, "neural_network_loss_curve.png"))
            plt.close()

    # 6. Logistic Regression: Visualize coefficients
    if "LogisticRegression" in models:
        lr_model = models["LogisticRegression"]
        if hasattr(lr_model, "coef_"):
            coef = lr_model.coef_[0]
            feature_names = list(X.columns)
            plt.figure(figsize=(10, 5))
            sns.barplot(x=feature_names, y=coef, palette="crest")
            plt.title("Logistic Regression Coefficients")
            plt.ylabel("Coefficient Value")
            plt.xlabel("Feature")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, "logistic_regression_coefficients.png"))
            plt.close()

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
