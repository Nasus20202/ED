"""
Centralized configuration file for flight delay prediction models.
All parameters can be tweaked from this single file.
"""

import os

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Data paths
DATA_DIR = "../data/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"
CODE_TO_AIRPORT_FILE_NAME = f"{DATA_DIR}airports.csv"
CODE_TO_AIRLINE_FILE_NAME = f"{DATA_DIR}airlines.csv"

# Data processing - Per-script data size limits
DATA_SIZE_LIMITS = {
    "model_comparison": 50_000,     # Fast comparison of models
    "genetic_tuning": 1_000,      # More data for hyperparameter tuning
    "benchmark_analysis": 10_000,  # Comprehensive benchmarking
    "default": 50_000               # Default fallback
}

DELAYED_THRESHOLD = 15  # Minutes delay to classify as "DELAYED"
FILL_NA_VALUE = 0  # Value to fill missing data
RANDOM_STATE = 42  # For reproducible results
TEST_SIZE = 0.2  # Fraction of data for testing

# Feature selection
CATEGORIES = [
    "AIRLINE",
    "ORIGIN_AIRPORT", 
    "DESTINATION_AIRPORT",
    "DAY_OF_WEEK",
    "MONTH",
    "TAIL_NUMBER",
]

NUMERICAL = [
    "YEAR",
    "DAY", 
    "FLIGHT_NUMBER",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
]

TARGET = "DELAYED"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "LogisticRegression": {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "C": 1.0
    },
    "DecisionTree": {
        "random_state": RANDOM_STATE,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "uniform"
    },
    "NeuralNetwork": {
        "hidden_layer_sizes": (100, 50),
        "random_state": RANDOM_STATE,
        "max_iter": 500,
        "alpha": 0.0001
    }
}

# Models to compare/optimize
MODELS_TO_COMPARE = [
    "RandomForest",
    "LogisticRegression", 
    "DecisionTree",
    "KNN",
    "NeuralNetwork"
]

# =============================================================================
# GENETIC ALGORITHM CONFIGURATION
# =============================================================================

# GA Parameters
GA_CONFIG = {
    "generations": 20,
    "population_size": 20,
    "crossover_probability": 0.5,
    "mutation_probability": 0.2,
    "tournament_size": 3,
    "mutation_sigma": 0.1,
    "mutation_indpb": 0.2
}

# Parameter ranges for optimization
PARAM_RANGES = {
    "RandomForest": {
        "n_estimators": (10, 200),
        "max_depth": (3, 20), 
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10)
    },
    "LogisticRegression": {
        "C": (0.01, 100.0),
        "max_iter": (100, 2000)
    },
    "NeuralNetwork": {
        "hidden_layer_size_1": (50, 200),
        "hidden_layer_size_2": (20, 100),
        "alpha": (0.0001, 0.01),
        "max_iter": (200, 1000)
    }
}

# Data size options for GA optimization
DATA_SIZE_OPTIONS = [1000, 5000, 10000]

# Cross-validation settings
CV_FOLDS = 3
CV_SCORING = 'accuracy'

# =============================================================================
# BENCHMARK CONFIGURATION  
# =============================================================================

# Optimized parameters (from genetic tuning results)
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

# Benchmark scenarios for feature removal analysis
FEATURE_REMOVAL_SCENARIOS = [
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
    {"name": "Categories Only", "excluded_categories": [], "excluded_numerical": NUMERICAL},
    {"name": "Numerical Only", "excluded_categories": CATEGORIES, "excluded_numerical": []},
    
    # Minimal feature sets
    {"name": "Essential Only", "excluded_categories": ["YEAR", "DAY", "FLIGHT_NUMBER"], "excluded_numerical": ["YEAR", "DAY", "FLIGHT_NUMBER", "SCHEDULED_ARRIVAL"]},
]

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# File naming
OUTPUT_FILES = {
    "model_comparison": "model_comparison_results.json",
    "genetic_tuning": "genetic_tuning_results.json", 
    "benchmark_analysis": "benchmark_analysis_results.json",
    "benchmark_analysis_full": "benchmark_analysis_results_full.json"
}

# Reporting settings
REPORTING = {
    "precision_digits": 4,
    "classification_report_digits": 3,
    "progress_bar": True,
    "verbose": True
}

# =============================================================================
# VALIDATION AND UTILITIES
# =============================================================================

def get_data_size_limit(script_name):
    """Get data size limit for specific script"""
    return DATA_SIZE_LIMITS.get(script_name, DATA_SIZE_LIMITS["default"])

def set_data_size_limit(script_name, limit):
    """Set data size limit for specific script"""
    DATA_SIZE_LIMITS[script_name] = limit

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Check data paths exist
    if not os.path.exists(DATA_DIR):
        errors.append(f"Data directory does not exist: {DATA_DIR}")
    
    # Check parameter consistency
    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        errors.append(f"TEST_SIZE must be between 0 and 1, got {TEST_SIZE}")
    
    if DELAYED_THRESHOLD < 0:
        errors.append(f"DELAYED_THRESHOLD must be non-negative, got {DELAYED_THRESHOLD}")
    
    # Check data size limits
    for script, limit in DATA_SIZE_LIMITS.items():
        if script != "default" and limit <= 0:
            errors.append(f"Data size limit for {script} must be positive, got {limit}")
    
    # Check GA parameters
    if GA_CONFIG["generations"] <= 0:
        errors.append("GA generations must be positive")
    
    if GA_CONFIG["population_size"] <= 0:
        errors.append("GA population_size must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

def get_model_list():
    """Get list of available models"""
    return list(DEFAULT_MODEL_PARAMS.keys())

def get_optimization_models():
    """Get list of models available for optimization"""
    return list(PARAM_RANGES.keys())

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed!")
