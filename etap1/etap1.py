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
DATA_DIR = "../data/"
IMG_DIR = "img/"
MAIN_DATA_FILE_NAME = f"{DATA_DIR}flights.csv"
CODE_TO_AIRPORT_FILE_NAME = f"{DATA_DIR}airports.csv"
CODE_TO_AIRLINE_FILE_NAME = f"{DATA_DIR}airlines.csv"
DEFAULT_BIN_SIZE = 30
SKIP_IF_IMAGES_EXIST = True  # Set to True to skip generation if images exist

main_data = pd.read_csv(MAIN_DATA_FILE_NAME)
code_to_airport = pd.read_csv(CODE_TO_AIRPORT_FILE_NAME)
code_to_airline = pd.read_csv(CODE_TO_AIRLINE_FILE_NAME)


def generate_histogram_image(
    column_name: str, column_data: np.ndarray, file_name: str
) -> None:
    if SKIP_IF_IMAGES_EXIST and os.path.exists(file_name):
        return
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
if not (
    SKIP_IF_IMAGES_EXIST
    and all(
        os.path.exists(f"{IMG_DIR}histogram_{col}.png") for col in main_data.columns
    )
):
    with Pool(processes=cpu_count()) as pool:
        for histogram_file_name in tqdm(
            pool.imap(process_column, columns_and_data),
            total=len(columns_and_data),
            desc="Generating histograms",
        ):
            histogram_file_names.append(histogram_file_name)
else:
    histogram_file_names = [
        f"{IMG_DIR}histogram_{col}.png" for col in main_data.columns
    ]

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


if not (SKIP_IF_IMAGES_EXIST and os.path.exists(f"{IMG_DIR}correlation_matrix.png")):
    print("Generating correlation matrix...")
    correlation_matrix = generate_correlation_matrix(main_data)
    correlation_matrix_file_name = f"{IMG_DIR}correlation_matrix.png"
    plot_correlation_matrix(correlation_matrix, correlation_matrix_file_name)
    generate_typst_image_element("Correlation Matrix", correlation_matrix_file_name)
else:
    correlation_matrix_file_name = f"{IMG_DIR}correlation_matrix.png"

with open(TYPST_CORRELATION_MATRIX_FILE_NAME, "w") as f:
    f.write(f'#image("{correlation_matrix_file_name}")\n')


## wygenerowaniue wykresu korelacji spearmana względem atrybutu ARRIVAL_DELAY bar plot
def plot_correlation_with_target(
    correlation_matrix: pd.DataFrame,
    target_column: str,
    file_name: str,
) -> None:
    # Exclude the target column from the plot
    corr = correlation_matrix[target_column].drop(
        labels=[target_column], errors="ignore"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(
        y=corr.index,
        x=corr.values,
        palette="coolwarm",
    )
    plt.title(f"Correlation with {target_column}")
    plt.xticks(rotation=90)
    plt.savefig(file_name)
    plt.close()


print("Generating correlation with ARRIVAL_DELAY...")
correlation_with_target_file_name = f"{IMG_DIR}correlation_with_target.png"
if not (SKIP_IF_IMAGES_EXIST and os.path.exists(correlation_with_target_file_name)):
    correlation_matrix = generate_correlation_matrix(main_data)
    plot_correlation_with_target(
        correlation_matrix, "ARRIVAL_DELAY", correlation_with_target_file_name
    )

    generate_typst_image_element(
        "Correlation with ARRIVAL_DELAY", correlation_with_target_file_name
    )
    with open("correlation_with_target.typ", "w") as f:
        f.write(f'#image("{correlation_with_target_file_name}")\n')


### Wyegenerowanie wykresów pudełkowych dla wszystkich atrybutów, podanie w opsicie przedzału najczęściej występujących wartości i liczby punktów oddaloncyh i mediany
def generate_boxplot(
    column_name: str,
    column_data: np.ndarray,
    file_name: str,
    median=None,
    outliers_count=None,
) -> None:
    # Only plot for numeric columns
    arr = np.asarray(column_data)
    arr = arr[~pd.isnull(arr)]
    if not np.issubdtype(arr.dtype, np.number):
        return  # Skip non-numeric columns
    if SKIP_IF_IMAGES_EXIST and os.path.exists(file_name):
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=arr)
    title = f"Boxplot of {column_name}"
    if median is not None and outliers_count is not None:
        title += f" (median={median}, outliers={outliers_count})"
    plt.title(title)
    plt.xlabel(column_name)
    plt.savefig(file_name)
    plt.close()


def calculate_boxplot_stats(arr: np.ndarray) -> tuple:
    arr = arr[~pd.isnull(arr)]
    if arr.size == 0:
        return (None, 0)
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
    return (median, len(outliers))


print("Generating boxplots...")


def process_boxplot_column(args: Tuple[str, np.ndarray]) -> str:
    column_name, column_data = args
    boxplot_file_name = f"{IMG_DIR}boxplot_{column_name}.png"
    median, outliers_count = calculate_boxplot_stats(column_data)
    generate_boxplot(
        column_name, column_data, boxplot_file_name, median, outliers_count
    )
    return boxplot_file_name


# Only process numeric columns for boxplots
numeric_columns_and_data: List[Tuple[str, np.ndarray]] = [
    (col, np.asarray(main_data[col]))
    for col in main_data.select_dtypes(include=[np.number]).columns
]

boxplot_file_names: List[str] = []
if not (
    SKIP_IF_IMAGES_EXIST
    and all(
        os.path.exists(f"{IMG_DIR}boxplot_{col}.png")
        for col, _ in numeric_columns_and_data
    )
):
    with Pool(processes=cpu_count()) as pool:
        for boxplot_file_name in tqdm(
            pool.imap(process_boxplot_column, numeric_columns_and_data),
            total=len(numeric_columns_and_data),
            desc="Generating boxplots",
        ):
            boxplot_file_names.append(boxplot_file_name)
else:
    boxplot_file_names = [
        f"{IMG_DIR}boxplot_{col}.png" for col, _ in numeric_columns_and_data
    ]

boxplot_image_elements: List[str] = []
for (col, col_data), boxplot_file_name in zip(
    numeric_columns_and_data, boxplot_file_names
):
    boxplot_image_element = generate_typst_image_element(col, boxplot_file_name)
    boxplot_image_elements.append(boxplot_image_element)

with open("boxplots.typ", "w") as f:
    f.write("#grid(\n" + "  columns: (1fr, 1fr),\n" + "  gutter: 3pt,")
    for i, boxplot_image_element in enumerate(boxplot_image_elements):
        f.write(
            "  "
            + boxplot_image_element
            + (",\n" if i != len(boxplot_image_elements) - 1 else "\n")
        )
    f.write("\n)")
    f.write("\n")
