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

# Import centralized configuration
from config import *

def load_and_prepare_data(data_size_limit):
    """Load and prepare flight data with specified size limit"""
    main_data = pd.read_csv(MAIN_DATA_FILE_NAME).head(data_size_limit)
    main_data[TARGET] = (main_data["ARRIVAL_DELAY"] > DELAYED_THRESHOLD).astype(int)
    
    # Fill missing values and filter
    main_data = main_data.fillna(FILL_NA_VALUE).dropna(subset=[TARGET])
    
    # Balance dataset
    delayed_flights = main_data[main_data[TARGET] == 1]
    non_delayed_flights = main_data[main_data[TARGET] == 0]
    min_samples = min(len(delayed_flights), len(non_delayed_flights))
    
    balanced_delayed = delayed_flights.sample(n=min_samples, random_state=RANDOM_STATE)
    balanced_non_delayed = non_delayed_flights.sample(n=min_samples, random_state=RANDOM_STATE)
    main_data = pd.concat([balanced_delayed, balanced_non_delayed]).reset_index(drop=True)
    
    # Encode categorical variables
    X = main_data[CATEGORIES + NUMERICAL].copy()
    for col in CATEGORIES:
        X[col] = X[col].astype("category").cat.codes
    
    y = main_data[TARGET]
    return X, y

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
    base_params = {"random_state": RANDOM_STATE}
    base_params.update(params)
    
    if model_type == "RandomForest":
        return RandomForestClassifier(**base_params)
    elif model_type == "LogisticRegression":
        return LogisticRegression(**base_params)
    elif model_type == "NeuralNetwork":
        return MLPClassifier(**base_params)

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
        scores = cross_val_score(model, X_sample, y_sample, cv=CV_FOLDS, scoring=CV_SCORING)
        return (scores.mean(),)
        
    except Exception as e:
        return (0.0,)

def optimize_model(model_type, X, y):
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
    toolbox.register("mutate", tools.mutGaussian, 
                    mu=0, sigma=GA_CONFIG["mutation_sigma"], indpb=GA_CONFIG["mutation_indpb"])
    toolbox.register("select", tools.selTournament, tournsize=GA_CONFIG["tournament_size"])
    
    # Run genetic algorithm
    population = toolbox.population(n=GA_CONFIG["population_size"])
    
    for gen in tqdm(range(GA_CONFIG["generations"]), desc=f"Evolving {model_type}"):
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG["crossover_probability"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < GA_CONFIG["mutation_probability"]:
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
    # Validate configuration
    validate_config()
    
    print("Starting Genetic Algorithm Hyperparameter Tuning...")
    print(f"Configuration: {GA_CONFIG['generations']} generations, {GA_CONFIG['population_size']} population")
    
    # Load full dataset for optimization using script-specific limit
    genetic_data_limit = get_data_size_limit("genetic_tuning")
    X_full, y_full = load_and_prepare_data(genetic_data_limit)
    
    models_to_optimize = get_optimization_models()
    results = {}
    
    for model_type in models_to_optimize:
        best_params, best_data_size, best_score = optimize_model(model_type, X_full, y_full)
        
        results[model_type] = {
            "best_parameters": best_params,
            "best_data_size": best_data_size,
            "best_cv_score": float(best_score)
        }
        
        if REPORTING["verbose"]:
            print(f"\n{model_type} Results:")
            print(f"Best Parameters: {best_params}")
            print(f"Best Data Size: {best_data_size}")
            print(f"Best CV Score: {best_score:.{REPORTING['precision_digits']}f}")
    
    # Add configuration info to results
    results["configuration"] = {
        "ga_config": GA_CONFIG,
        "param_ranges": PARAM_RANGES,
        "data_size_options": DATA_SIZE_OPTIONS,
        "data_size_limit": genetic_data_limit,
        "cv_folds": CV_FOLDS
    }
    
    # Save results
    with open(OUTPUT_FILES["genetic_tuning"], "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOptimization complete! Results saved to {OUTPUT_FILES['genetic_tuning']}")
    
    # Test best models
    print("\nTesting optimized models...")
    test_data_limit = min(genetic_data_limit, 100000)  # Use smaller subset for testing
    X_test, y_test = load_and_prepare_data(test_data_limit)
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