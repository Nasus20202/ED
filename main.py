import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Oprócz Państwa danych i danych ogólnych (imię, nazwisko, numer indeksu, data, nazwa
# sprawozdania), sprawozdanie powinno zawierać:
#  ogólny opis zbioru danych,
#  określenie celu eksploracji i kryteriów sukcesu,
#  charakterystyka zbioru danych (pochodzenie, format, liczba przykładów, czy składa się z
# jednego zbioru, czy z kilku)
#  opis atrybutów (nazwy, typy - nominalne/numeryczne, co oznaczają, co oznaczają wartości -
# jednostka miary, wartości specjalne),
#  wyniki eksploracyjnej analizy danych (EDA):
#  rozkłady wartości atrybutów,
#  korelacje pomiędzy wartościami atrybutów,
#  wstępne ustalenia dotyczące zawartości zbioru,
#  uwagi nt. jakości danych:
#  dane brakujące,
#  punkty oddalone,
#  dane niespójne,
#  dane niezrozumiałe,
#  opis wyników EDA w odniesieniu do celów eksploracji (czy dane są wystarczające),
#  ewentualna rewizja celów.

# 2015 Flight Delays and Cancellations

TYPST_HISTOGRAMS_FILE_NAME = "histograms.typ"
TYPST_CORRELATION_MATRIX_FILE_NAME = "correlation_matrix.typ"
DATA_DIR = "data/"
IMG_DIR = "img/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"
CODE_TO_AIRPORT_FILE_NAME = f"{DATA_DIR}airports.csv"
CODE_TO_AIRLINE_FILE_NAME = f"{DATA_DIR}airlines.csv"
DEFAULT_BIN_SIZE = 30

main_data = pd.read_csv(MAIN_DATA_FILE_NAME)
code_to_airport = pd.read_csv(CODE_TO_AIRPORT_FILE_NAME)
code_to_airline = pd.read_csv(CODE_TO_AIRLINE_FILE_NAME)


def generate_histogram_image(
    column_name: str, column_data: np.ndarray, file_name: str
) -> None:
    plt.figure(figsize=(10, 6))
    arr = np.asarray(column_data)
    arr = arr[~pd.isnull(arr)]
    if np.issubdtype(arr.dtype, np.number):
        sns.histplot(arr, bins=DEFAULT_BIN_SIZE)
    else:
        sns.histplot(arr.astype(str))
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.savefig(file_name)
    plt.close()


def generate_typst_image_element(column_name: str, image_file_name: str) -> str:
    return f'image("{image_file_name}")'


os.makedirs(IMG_DIR, exist_ok=True)


def process_column(args: Tuple[str, np.ndarray]) -> str:
    column_name, column_data = args
    histogram_file_name = f"{IMG_DIR}histogram_{column_name}.png"
    generate_histogram_image(column_name, column_data, histogram_file_name)
    return histogram_file_name


columns_and_data: List[Tuple[str, np.ndarray]] = [
    (col, np.asarray(main_data[col])) for col in main_data.columns
]

histogram_file_names: List[str] = []
with Pool(processes=cpu_count()) as pool:
    for histogram_file_name in tqdm(
        pool.imap(process_column, columns_and_data),
        total=len(columns_and_data),
        desc="Generating histograms",
    ):
        histogram_file_names.append(histogram_file_name)


typst_image_elements: List[str] = []
for histogram_file_name in histogram_file_names:
    typst_image_element = generate_typst_image_element(
        histogram_file_name, histogram_file_name
    )
    typst_image_elements.append(typst_image_element)


with open(TYPST_HISTOGRAMS_FILE_NAME, "w") as f:
    f.write("#grid(\n" + "  columns: (1fr, 1fr),\n" + "  gutter: 3pt,")

    for i, typst_image_element in enumerate(typst_image_elements):
        f.write(
            "  "
            + typst_image_element
            + (",\n" if i != len(typst_image_elements) - 1 else "\n")
        )

    f.write("\n)")


# Wgenerowanie mecierzy korealcji metodą Spearmana
def generate_correlation_matrix(
    data: pd.DataFrame, method: str = "spearman"
) -> pd.DataFrame:
    return data.corr(method=method, numeric_only=True)


def plot_correlation_matrix(correlation_matrix: pd.DataFrame, file_name: str) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig(file_name)
    plt.close()


correlation_matrix = generate_correlation_matrix(main_data)
correlation_matrix_file_name = f"{IMG_DIR}correlation_matrix.png"
plot_correlation_matrix(correlation_matrix, correlation_matrix_file_name)
generate_typst_image_element("Correlation Matrix", correlation_matrix_file_name)
with open(TYPST_CORRELATION_MATRIX_FILE_NAME, "w") as f:
    f.write(f'#image("{correlation_matrix_file_name}")\n')


## wygenerowaniue wykresu korelacji względem atrybutu ARRIVAL_DELAY
