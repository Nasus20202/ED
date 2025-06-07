import pandas as pd
import json
import argparse
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Hardcoded best parameters from genetic optimization
BEST_PARAMS = {
    "RandomForest": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "LogisticRegression": {
        "C": 10.0,
        "max_iter": 1500
    },
    "NeuralNetwork": {
        "hidden_layer_sizes": (150, 75),
        "alpha": 0.001,
        "max_iter": 800
    }
}

# Load and prepare data
DATA_DIR = "../data/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"

def get_dataset_size():
    """Get the total size of the dataset"""
    df = pd.read_csv(MAIN_DATA_FILE_NAME)
    return len(df)

def load_and_prepare_data(data_size_limit, excluded_categories=None, excluded_numerical=None):
    """Load and prepare flight data with optional feature exclusion"""
    if data_size_limit == -1:  # Use full dataset
        main_data = pd.read_csv(MAIN_DATA_FILE_NAME)
        print(f"Using full dataset: {len(main_data):,} rows")
    else:
        main_data = pd.read_csv(MAIN_DATA_FILE_NAME).head(data_size_limit)
        print(f"Using limited dataset: {len(main_data):,} rows")
    
    main_data["DELAYED"] = (main_data["ARRIVAL_DELAY"] > 15).astype(int)
    TARGET = "DELAYED"
    
    all_categories = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK", "MONTH", "TAIL_NUMBER"]
    all_numerical = ["YEAR", "DAY", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL"]
    
    # Remove excluded features
    categories = [col for col in all_categories if col not in (excluded_categories or [])]
    numerical = [col for col in all_numerical if col not in (excluded_numerical or [])]
    
    # Fill missing values and filter
    main_data = main_data.fillna(0).dropna(subset=[TARGET])
    
    # Balance dataset
    delayed_flights = main_data[main_data[TARGET] == 1]
    non_delayed_flights = main_data[main_data[TARGET] == 0]
    min_samples = min(len(delayed_flights), len(non_delayed_flights))
    
    balanced_delayed = delayed_flights.sample(n=min_samples, random_state=42)
    balanced_non_delayed = non_delayed_flights.sample(n=min_samples, random_state=42)
    main_data = pd.concat([balanced_delayed, balanced_non_delayed]).reset_index(drop=True)
    
    # Encode categorical variables
    X = main_data[categories + numerical].copy()
    for col in categories:
        X[col] = X[col].astype("category").cat.codes
    
    y = main_data[TARGET]
    return X, y, categories, numerical

def create_model(model_type, params):
    """Create model instance with given parameters"""
    if model_type == "RandomForest":
        return RandomForestClassifier(random_state=42, **params)
    elif model_type == "LogisticRegression":
        return LogisticRegression(random_state=42, **params)
    elif model_type == "NeuralNetwork":
        return MLPClassifier(random_state=42, **params)

def benchmark_baseline(data_size_limit):
    """Benchmark models with all features"""
    print("=== BASELINE BENCHMARK (All Features) ===")
    X, y, categories, numerical = load_and_prepare_data(data_size_limit)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    for model_name, params in tqdm(BEST_PARAMS.items(), desc="Baseline models"):
        model = create_model(model_name, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[model_name] = {
            "accuracy": acc,
            "features_used": len(categories + numerical),
            "categories": categories,
            "numerical": numerical
        }
        
        print(f"\n{model_name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Features used: {len(categories + numerical)}")
    
    return results

def benchmark_feature_removal(data_size_limit):
    """Benchmark models with systematic feature removal"""
    print("\n=== FEATURE REMOVAL ANALYSIS ===")
    
    all_categories = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK", "MONTH"]
    all_numerical = ["YEAR", "DAY", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL"]
    
    removal_scenarios = [
        # Remove single category features
        {"name": "No AIRLINE", "excluded_categories": ["AIRLINE"], "excluded_numerical": []},
        {"name": "No ORIGIN_AIRPORT", "excluded_categories": ["ORIGIN_AIRPORT"], "excluded_numerical": []},
        {"name": "No DESTINATION_AIRPORT", "excluded_categories": ["DESTINATION_AIRPORT"], "excluded_numerical": []},
        {"name": "No DAY_OF_WEEK", "excluded_categories": ["DAY_OF_WEEK"], "excluded_numerical": []},
        {"name": "No MONTH", "excluded_categories": ["MONTH"], "excluded_numerical": []},
        
        # Remove single numerical features
        {"name": "No DISTANCE", "excluded_categories": [], "excluded_numerical": ["DISTANCE"]},
        {"name": "No SCHEDULED_DEPARTURE", "excluded_categories": [], "excluded_numerical": ["SCHEDULED_DEPARTURE"]},
        {"name": "No SCHEDULED_TIME", "excluded_categories": [], "excluded_numerical": ["SCHEDULED_TIME"]},
        
        # Remove multiple features
        {"name": "No Airport Info", "excluded_categories": ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], "excluded_numerical": []},
        {"name": "No Time Info", "excluded_categories": ["DAY_OF_WEEK", "MONTH"], "excluded_numerical": ["SCHEDULED_DEPARTURE", "SCHEDULED_TIME"]},
        {"name": "Categories Only", "excluded_categories": [], "excluded_numerical": all_numerical},
        {"name": "Numerical Only", "excluded_categories": all_categories, "excluded_numerical": []},
        
        # Minimal feature sets
        {"name": "Essential Only", "excluded_categories": ["YEAR", "DAY", "FLIGHT_NUMBER"], "excluded_numerical": ["YEAR", "DAY", "FLIGHT_NUMBER", "SCHEDULED_ARRIVAL"]},
    ]
    
    feature_analysis_results = {}
    
    for scenario in tqdm(removal_scenarios, desc="Testing scenarios"):
        scenario_name = scenario["name"]
        excluded_cat = scenario["excluded_categories"]
        excluded_num = scenario["excluded_numerical"]
        
        try:
            X, y, categories, numerical = load_and_prepare_data(
                data_size_limit, excluded_cat, excluded_num
            )
            
            if len(categories) + len(numerical) == 0:
                print(f"Skipping {scenario_name}: No features remaining")
                continue
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scenario_results = {}
            
            for model_name, params in BEST_PARAMS.items():
                try:
                    model = create_model(model_name, params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    
                    scenario_results[model_name] = {
                        "accuracy": acc,
                        "features_used": len(categories + numerical),
                        "categories_used": categories,
                        "numerical_used": numerical
                    }
                    
                except Exception as e:
                    scenario_results[model_name] = {
                        "accuracy": 0.0,
                        "error": str(e),
                        "features_used": len(categories + numerical)
                    }
            
            feature_analysis_results[scenario_name] = scenario_results
            
        except Exception as e:
            print(f"Error in scenario {scenario_name}: {str(e)}")
            continue
    
    return feature_analysis_results

def analyze_feature_importance(data_size_limit):
    """Analyze which features are most important using Random Forest"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    X, y, categories, numerical = load_and_prepare_data(data_size_limit)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Random Forest for feature importance
    rf_model = create_model("RandomForest", BEST_PARAMS["RandomForest"])
    rf_model.fit(X_train, y_train)
    
    feature_names = categories + numerical
    importances = rf_model.feature_importances_
    
    # Sort features by importance
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature Importance Ranking:")
    for i, (feature, importance) in enumerate(feature_importance):
        print(f"{i+1:2d}. {feature:20s}: {importance:.4f}")
    
    return feature_importance

def create_performance_summary(baseline_results, feature_analysis_results):
    """Create comprehensive performance summary"""
    print("\n=== PERFORMANCE SUMMARY ===")
    
    # Baseline performance
    print("\nBaseline Performance (All Features):")
    for model, results in baseline_results.items():
        print(f"{model:20s}: {results['accuracy']:.4f}")
    
    # Feature removal impact
    print("\nFeature Removal Impact:")
    print(f"{'Scenario':<20s} {'RF':<8s} {'LR':<8s} {'NN':<8s} {'Features':<10s}")
    print("-" * 60)
    
    for scenario, results in feature_analysis_results.items():
        rf_acc = results.get("RandomForest", {}).get("accuracy", 0.0)
        lr_acc = results.get("LogisticRegression", {}).get("accuracy", 0.0)
        nn_acc = results.get("NeuralNetwork", {}).get("accuracy", 0.0)
        features = results.get("RandomForest", {}).get("features_used", 0)
        
        print(f"{scenario:<20s} {rf_acc:<8.4f} {lr_acc:<8.4f} {nn_acc:<8.4f} {features:<10d}")
    
    # Best and worst scenarios
    print("\nBest Performing Scenarios:")
    for model_name in ["RandomForest", "LogisticRegression", "NeuralNetwork"]:
        best_scenario = max(
            feature_analysis_results.items(),
            key=lambda x: x[1].get(model_name, {}).get("accuracy", 0.0)
        )
        best_acc = best_scenario[1].get(model_name, {}).get("accuracy", 0.0)
        print(f"{model_name}: {best_scenario[0]} ({best_acc:.4f})")

def save_results(baseline_results, feature_analysis_results, feature_importance, data_size_limit):
    """Save all benchmark results to JSON file"""
    all_results = {
        "baseline_results": baseline_results,
        "feature_analysis_results": feature_analysis_results,
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in feature_importance],
        "metadata": {
            "data_size_limit": data_size_limit,
            "actual_data_size": "full" if data_size_limit == -1 else data_size_limit,
            "best_parameters_used": BEST_PARAMS
        }
    }
    
    filename = "benchmark_analysis_results_full.json" if data_size_limit == -1 else "benchmark_analysis_results.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description="Benchmark flight delay prediction models")
    parser.add_argument("--full", action="store_true", help="Use full dataset instead of limited size")
    parser.add_argument("--size", type=int, default=10000, help="Dataset size limit (default: 10000)")
    
    args = parser.parse_args()
    
    if args.full:
        data_size_limit = -1
        total_size = get_dataset_size()
        print(f"Using FULL dataset ({total_size:,} rows)")
    else:
        data_size_limit = args.size
        print(f"Using LIMITED dataset ({data_size_limit:,} rows)")
    
    print("Starting Comprehensive Model Benchmarking...")
    print(f"Models to test: {list(BEST_PARAMS.keys())}")
    
    # Run baseline benchmark
    baseline_results = benchmark_baseline(data_size_limit)
    
    # Run feature removal analysis
    feature_analysis_results = benchmark_feature_removal(data_size_limit)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(data_size_limit)
    
    # Create summary
    create_performance_summary(baseline_results, feature_analysis_results)
    
    # Save results
    save_results(baseline_results, feature_analysis_results, feature_importance, data_size_limit)
    
    print("\nBenchmarking complete!")
    
    if data_size_limit == -1:
        print("Full dataset benchmarking completed!")
    else:
        print(f"To run on full dataset, use: python benchmark_analysis.py --full")

if __name__ == "__main__":
    main()
