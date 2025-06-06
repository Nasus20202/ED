import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier


# 2015 flight delays and cancellations dataset

#YEAR,MONTH,DAY,DAY_OF_WEEK,AIRLINE,FLIGHT_NUMBER,TAIL_NUMBER,ORIGIN_AIRPORT,DESTINATION_AIRPORT,SCHEDULED_DEPARTURE,DEPARTURE_TIME,DEPARTURE_DELAY,TAXI_OUT,WHEELS_OFF,SCHEDULED_TIME,ELAPSED_TIME,AIR_TIME,DISTANCE,WHEELS_ON,TAXI_IN,SCHEDULED_ARRIVAL,ARRIVAL_TIME,ARRIVAL_DELAY,DIVERTED,CANCELLED,CANCELLATION_REASON,AIR_SYSTEM_DELAY,SECURITY_DELAY,AIRLINE_DELAY,LATE_AIRCRAFT_DELAY,WEATHER_DELAY
# 2015,11,24,2,DL,957,N959DL,TLH,ATL,0700,0742,42,29,0811,65,82,44,223,0855,9,0805,0904,59,0,0,,17,0,42,0,0

DATA_DIR = "data/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"
CODE_TO_AIRPORT_FILE_NAME = f"{DATA_DIR}airports.csv"
CODE_TO_AIRLINE_FILE_NAME = f"{DATA_DIR}airlines.csv"


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
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

# Print summary table
print("\nSummary of Model Accuracies:")
for name, acc in results.items():
    print(f"{name:15}: {acc:.4f}")


