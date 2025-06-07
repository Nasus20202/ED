import pandas as pd
import json
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# 2015 flight delays and cancellations dataset

#YEAR,MONTH,DAY,DAY_OF_WEEK,AIRLINE,FLIGHT_NUMBER,TAIL_NUMBER,ORIGIN_AIRPORT,DESTINATION_AIRPORT,SCHEDULED_DEPARTURE,DEPARTURE_TIME,DEPARTURE_DELAY,TAXI_OUT,WHEELS_OFF,SCHEDULED_TIME,ELAPSED_TIME,AIR_TIME,DISTANCE,WHEELS_ON,TAXI_IN,SCHEDULED_ARRIVAL,ARRIVAL_TIME,ARRIVAL_DELAY,DIVERTED,CANCELLED,CANCELLATION_REASON,AIR_SYSTEM_DELAY,SECURITY_DELAY,AIRLINE_DELAY,LATE_AIRCRAFT_DELAY,WEATHER_DELAY
# 2015,11,24,2,DL,957,N959DL,TLH,ATL,0700,0742,42,29,0811,65,82,44,223,0855,9,0805,0904,59,0,0,,17,0,42,0,0

DATA_DIR = "../data/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"
CODE_TO_AIRPORT_FILE_NAME = f"{DATA_DIR}airports.csv"
CODE_TO_AIRLINE_FILE_NAME = f"{DATA_DIR}airlines.csv"
DATA_SIZE_LIMIT = 50_000

main_data = pd.read_csv(MAIN_DATA_FILE_NAME)
code_to_airport = pd.read_csv(CODE_TO_AIRPORT_FILE_NAME)
code_to_airline = pd.read_csv(CODE_TO_AIRLINE_FILE_NAME)

# main_data = main_data.dropna()
main_data["DELAYED"] = (main_data["ARRIVAL_DELAY"] > 15).astype(int)
TARGET = "DELAYED"

categories = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DAY_OF_WEEK",
    "MONTH",
    "TAIL_NUMBER",
]

# Remove features that are known only AFTER the flight (data leakage)
numerical = [
    "YEAR",
    "DAY",
    "FLIGHT_NUMBER",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
    # Removed: DEPARTURE_TIME, DEPARTURE_DELAY, TAXI_OUT, WHEELS_OFF, 
    # ELAPSED_TIME, AIR_TIME, WHEELS_ON, TAXI_IN, ARRIVAL_TIME, ARRIVAL_DELAY
    # These are all known only after/during the flight
]

# --- New code for testing prediction methods ---

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Fill missing values for simplicity (could be improved)
main_data = main_data.fillna(0)

# Filter out rows where target is missing
main_data = main_data.dropna(subset=[TARGET])

# Balance the dataset - equal amounts of delayed and non-delayed flights
delayed_flights = main_data[main_data[TARGET] == 1]
non_delayed_flights = main_data[main_data[TARGET] == 0]

print(f"Original delayed flights: {len(delayed_flights)}")
print(f"Original non-delayed flights: {len(non_delayed_flights)}")

# Take minimum of both classes to balance
min_samples = min(min(len(delayed_flights), len(non_delayed_flights)), DATA_SIZE_LIMIT)
balanced_delayed = delayed_flights.sample(n=min_samples, random_state=42)
balanced_non_delayed = non_delayed_flights.sample(n=min_samples, random_state=42)

# Combine balanced data
main_data = pd.concat([balanced_delayed, balanced_non_delayed]).reset_index(drop=True)

print(f"Balanced delayed flights: {(main_data[TARGET] == 1).sum()}")
print(f"Balanced non-delayed flights: {(main_data[TARGET] == 0).sum()}")

# Encode categorical variables
X = main_data[categories + numerical].copy()
for col in categories:
    X[col] = X[col].astype("category").cat.codes

y = main_data[TARGET]

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
}

results = {}
detailed_results = {}

for name, model in models.items():
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
    
    tqdm.write(f"{name} Results:")
    tqdm.write(f"Accuracy: {acc:.4f}")
    tqdm.write(classification_report(y_test, y_pred, digits=3))

# Print summary table
tqdm.write("\nSummary of Model Accuracies:")
for name, acc in results.items():
    tqdm.write(f"{name:15}: {acc:.4f}")

# Save results to file
results_file = "model_comparison_results.json"
save_data = {
    "dataset_info": {
        "shape": X.shape,
        "target_distribution": y.value_counts().to_dict(),
        "test_size": 0.2
    },
    "model_results": detailed_results,
    "summary": results
}

with open(results_file, 'w') as f:
    json.dump(save_data, f, indent=2)

tqdm.write(f"\nResults saved to {results_file}")
