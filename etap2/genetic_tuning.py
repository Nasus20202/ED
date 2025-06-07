import pandas as pd
import numpy as np
import json
import random
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
DATA_DIR = "../data/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"

def load_and_prepare_data(data_size_limit):
    """Load and prepare flight data with specified size limit"""
    main_data = pd.read_csv(MAIN_DATA_FILE_NAME).head(data_size_limit)
    main_data["DELAYED"] = (main_data["ARRIVAL_DELAY"] > 15).astype(int)
    TARGET = "DELAYED"
    
    categories = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK", "MONTH", "TAIL_NUMBER"]
    numerical = ["YEAR", "DAY", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL"]
    
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
    return X, y

# Define parameter ranges for each model
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

# Data size options
DATA_SIZE_OPTIONS = [1000]

def decode_individual(individual, model_type):
    """Decode genetic algorithm individual to model parameters"""
    params = {}
    ranges = PARAM_RANGES[model_type]
    
    if model_type == "RandomForest":
        params["n_estimators"] = int(individual[0] * (ranges["n_estimators"][1] - ranges["n_estimators"][0]) + ranges["n_estimators"][0])
        params["max_depth"] = int(individual[1] * (ranges["max_depth"][1] - ranges["max_depth"][0]) + ranges["max_depth"][0])
        params["min_samples_split"] = int(individual[2] * (ranges["min_samples_split"][1] - ranges["min_samples_split"][0]) + ranges["min_samples_split"][0])
        params["min_samples_leaf"] = int(individual[3] * (ranges["min_samples_leaf"][1] - ranges["min_samples_leaf"][0]) + ranges["min_samples_leaf"][0])
        
    elif model_type == "LogisticRegression":
        params["C"] = individual[0] * (ranges["C"][1] - ranges["C"][0]) + ranges["C"][0]
        params["max_iter"] = int(individual[1] * (ranges["max_iter"][1] - ranges["max_iter"][0]) + ranges["max_iter"][0])
        
    elif model_type == "NeuralNetwork":
        h1 = int(individual[0] * (ranges["hidden_layer_size_1"][1] - ranges["hidden_layer_size_1"][0]) + ranges["hidden_layer_size_1"][0])
        h2 = int(individual[1] * (ranges["hidden_layer_size_2"][1] - ranges["hidden_layer_size_2"][0]) + ranges["hidden_layer_size_2"][0])
        params["hidden_layer_sizes"] = (h1, h2)
        params["alpha"] = individual[2] * (ranges["alpha"][1] - ranges["alpha"][0]) + ranges["alpha"][0]
        params["max_iter"] = int(individual[3] * (ranges["max_iter"][1] - ranges["max_iter"][0]) + ranges["max_iter"][0])
    
    return params

def create_model(model_type, params):
    """Create model instance with given parameters"""
    if model_type == "RandomForest":
        return RandomForestClassifier(random_state=42, **params)
    elif model_type == "LogisticRegression":
        return LogisticRegression(random_state=42, **params)
    elif model_type == "NeuralNetwork":
        return MLPClassifier(random_state=42, **params)

def evaluate_individual(individual, model_type, X, y):
    """Evaluate individual using cross-validation"""
    try:
        # Decode data size (last gene)
        data_size_idx = int(individual[-1] * len(DATA_SIZE_OPTIONS))
        data_size_idx = min(data_size_idx, len(DATA_SIZE_OPTIONS) - 1)
        data_size = DATA_SIZE_OPTIONS[data_size_idx]
        
        # Sample data if needed
        if len(X) > data_size:
            indices = np.random.choice(len(X), data_size, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample, y_sample = X, y
        
        # Decode parameters
        params = decode_individual(individual[:-1], model_type)
        model = create_model(model_type, params)
        
        # Cross-validation
        scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring='accuracy')
        return (scores.mean(),)
        
    except Exception as e:
        return (0.0,)

def optimize_model(model_type, X, y, generations=20, population_size=20):
    """Optimize model hyperparameters using genetic algorithm"""
    print(f"\nOptimizing {model_type}...")
    
    # Determine chromosome length
    param_count = len(PARAM_RANGES[model_type])
    chromosome_length = param_count + 1  # +1 for data size
    
    # Setup genetic algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, chromosome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, model_type=model_type, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run genetic algorithm
    population = toolbox.population(n=population_size)
    
    for gen in tqdm(range(generations), desc=f"Evolving {model_type}"):
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Clip values to [0, 1]
        for ind in offspring:
            for i in range(len(ind)):
                ind[i] = max(0, min(1, ind[i]))
        
        population[:] = offspring
    
    # Get best individual
    best_ind = tools.selBest(population, 1)[0]
    best_params = decode_individual(best_ind[:-1], model_type)
    
    data_size_idx = int(best_ind[-1] * len(DATA_SIZE_OPTIONS))
    data_size_idx = min(data_size_idx, len(DATA_SIZE_OPTIONS) - 1)
    best_data_size = DATA_SIZE_OPTIONS[data_size_idx]
    
    return best_params, best_data_size, best_ind.fitness.values[0]

def main():
    """Main optimization function"""
    print("Starting Genetic Algorithm Hyperparameter Tuning...")
    
    # Load full dataset for optimization
    X_full, y_full = load_and_prepare_data(300000)
    
    models_to_optimize = ["RandomForest", "LogisticRegression", "NeuralNetwork"]
    results = {}
    
    for model_type in models_to_optimize:
        best_params, best_data_size, best_score = optimize_model(
            model_type, X_full, y_full, generations=15, population_size=15
        )
        
        results[model_type] = {
            "best_parameters": best_params,
            "best_data_size": best_data_size,
            "best_cv_score": float(best_score)
        }
        
        print(f"\n{model_type} Results:")
        print(f"Best Parameters: {best_params}")
        print(f"Best Data Size: {best_data_size}")
        print(f"Best CV Score: {best_score:.4f}")
    
    # Save results
    with open("genetic_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOptimization complete! Results saved to genetic_tuning_results.json")
    
    # Test best models
    print("\nTesting optimized models...")
    X_test, y_test = load_and_prepare_data(100000)
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
    
    test_results = {}
    for model_type, result in results.items():
        model = create_model(model_type, result["best_parameters"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        test_results[model_type] = acc
        print(f"{model_type}: {acc:.4f}")
    
    print(f"\nBest performing model: {max(test_results, key=test_results.get)} ({max(test_results.values()):.4f})")

if __name__ == "__main__":
    main()
