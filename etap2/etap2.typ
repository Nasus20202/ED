#set text(
  font: "New Computer Modern",
  size: 12pt,
  lang: "pl"
)
#set page(
  paper: "a4",
  margin: (x: 1cm, y: 2cm),
  numbering: "1",
  header: [Eksploracja danych - etap 2 #line(length: 100%)],
)
#set heading(numbering: "1.")
#show link: underline

#set table(
  stroke: none,
  fill: (x, y) => if y == 0 {
    gray
  } else if calc.rem(y, 2) == 0 {
    silver
  },
)

#align(center)[
  #stack(
    v(12pt),
    text(size: 20pt)[Eksploracja danych - etap 2],
    v(12pt),
    text(size: 15pt)[Krzysztof Nasuta 193328, Filip Dawidowski 193433, Aleks Iwicki 193354],
  )
]

= Charakterystyka zbioru
- Pochodzenie: #link("https://www.kaggle.com/datasets/usdot/flight-delays/data")[Kaggle]
- Liczba przykładów: 5819080
- Format: CSV (3 pliki: `flights.csv` - właściwy zbiór, `airports.csv` - informacje o lotniskach, `airlines.csv` - informacje o liniach lotniczych)
- Ilość zbiorów danych: 1


= Wprowadzenie
Dataset: *2015 Flight Delays and Cancellations* \
Cel: Budowa modelu predykcyjnego klasyfikującego opóźnienia lotów (ARRIVAL_DELAY > 15 minut)

Opóźnienia lotów mają znaczący wpływ na funkcjonowanie transportu lotniczego. Niniejszy projekt ma na celu stworzenie modelu uczenia maszynowego przewidującego opóźnienia.

*Kluczowe pytania badawcze:*
- Które czynniki najsilniej wpływają na opóźnienia?
- Który algorytm osiąga najlepsze wyniki?


= Założenia wstępne
Podczas przewidywania opóźnień lotów nie będziemy uwzględniać informacji, które nie są dostępne w momencie planowania lotu, takich jak:
- DEPARTURE_TIME (nie mylić z SCHEDULED_DEPARTURE)
- DEPARTURE_DELAY
- TAXI_OUT
- WHEELS_OFF
- ELAPSED_TIME
- AIR_TIME
- WHEELS_ON
- TAXI_IN
- ARRIVAL_TIME
- ARRIVAL_DELAY

Spowoduje to znaczne obniżenie dokładności modeli, lecz pozwoli na realistyczne przewidywanie.


= Przygotowanie Danych
*Źródła danych:*
- flights.csv
- airlines.csv 
- airports.csv

*Kroki przetwarzania:*
1. Ładowanie danych z pliku flights.csv z ograniczeniem do 250,000 rekordów
2. Definicja zmiennej celu: `DELAYED = 1` jeśli `ARRIVAL_DELAY > 15`
4. Balansowanie zbioru danych - równa liczba opóźnionych i nieopóźnionych lotów
6. Podział zbalansowanych danych na zbiór treningowy (80%) i testowy (20%)

*Cechy wykorzystane w modelu:*
- Kategoryczne: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, DAY_OF_WEEK, MONTH
- Numeryczne: YEAR, DAY, FLIGHT_NUMBER, SCHEDULED_DEPARTURE, SCHEDULED_TIME, DISTANCE, SCHEDULED_ARRIVAL

*Usunięte cechy (data leakage):* DEPARTURE_TIME, DEPARTURE_DELAY, TAXI_OUT, WHEELS_OFF, ELAPSED_TIME, AIR_TIME, WHEELS_ON, TAXI_IN, ARRIVAL_TIME, ARRIVAL_DELAY

= Metodologia
*Wykorzystane modele:*

#table(
  columns: 2,
  [Model], [Implementacja],
  ["Drzewo Decyzyjne"], "DecisionTreeClassifier(random_state=42)",
  ["Las Losowy"], "RandomForestClassifier(n_estimators=100, random_state=42)",
  ["Regresja Logistyczna"], "LogisticRegression(random_state=42, max_iter=1000)",
  ["K-NN"], "KNeighborsClassifier(n_neighbors=5)",
  ["Sieć Neuronowa"], "MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500)",
)

*Optymalizacja hiperparametrów (Algorytm Genetyczny):*
- Populacja: 15 osobników
- Generacje: 15
- Krzyżowanie: dwupunktowe (prawdopodobieństwo 0.5)
- Mutacja: gaussowska (prawdopodobieństwo 0.2, σ=0.1)
- Selekcja: turniejowa (rozmiar turnieju = 3)
- Fitness: średnia dokładność z 3-krotnej walidacji krzyżowej

*Parametry optymalizowane:*

#table(
  columns: 3,
  [Model], [Parametr], [Zakres],
  ["Random Forest"], "n_estimators", "10-200",
  [], "max_depth", "3-20", 
  [], "min_samples_split", "2-20",
  [], "min_samples_leaf", "1-10",
  ["Logistic Regression"], "C", "0.01-100.0",
  [], "max_iter", "100-2000",
  ["Neural Network"], "hidden_layer_size_1", "50-200",
  [], "hidden_layer_size_2", "20-100",
  [], "alpha", "0.0001-0.01",
  [], "max_iter", "200-1000",
)

= Wyniki Eksperymentów

== Wyniki modeli z domyślnymi parametrami
*Dataset: `x` rekordów (zbalansowany)*

#table(
  columns: 3,
  [Model], [Dokładność], [Cechy],
  ["Decision Tree"], "`x`", "`x`",
  ["Random Forest"], "`x`", "`x`",
  ["Logistic Regression"], "`x`", "`x`",
  ["K-NN"], "`x`", "`x`",
  ["Neural Network"], "`x`", "`x`",
)

== Wyniki optymalizacji genetycznej
*Najlepsze parametry znalezione przez algorytm genetyczny:*

#table(
  columns: 4,
  [Model], [Najlepsze parametry], [CV Score], [Rozmiar danych],
  ["Random Forest"], "`x`", "`x`", "`x`",
  ["Logistic Regression"], "`x`", "`x`", "`x`", 
  ["Neural Network"], "`x`", "`x`", "`x`",
)

== Analiza ważności cech
*Ranking ważności cech (Random Forest):*

#table(
  columns: 2,
  [Pozycja], [Cecha], [Ważność],
  ["1."], "`x`", "`x`",
  ["2."], "`x`", "`x`",
  ["3."], "`x`", "`x`",
  ["4."], "`x`", "`x`",
  ["5."], "`x`", "`x`",
)

== Analiza wpływu usuwania cech
*Wpływ usunięcia poszczególnych cech na dokładność:*

#table(
  columns: 5,
  [Scenariusz], [Random Forest], [Logistic Regression], [Neural Network], [Liczba cech],
  ["Wszystkie cechy"], "`x`", "`x`", "`x`", "`x`",
  ["Bez AIRLINE"], "`x`", "`x`", "`x`", "`x`",
  ["Bez ORIGIN_AIRPORT"], "`x`", "`x`", "`x`", "`x`",
  ["Bez DESTINATION_AIRPORT"], "`x`", "`x`", "`x`", "`x`",
  ["Bez DISTANCE"], "`x`", "`x`", "`x`", "`x`",
  ["Bez informacji o lotnisku"], "`x`", "`x`", "`x`", "`x`",
  ["Bez informacji czasowych"], "`x`", "`x`", "`x`", "`x`",
  ["Tylko kategoryczne"], "`x`", "`x`", "`x`", "`x`",
  ["Tylko numeryczne"], "`x`", "`x`", "`x`", "`x`",
)

*Najlepsze scenariusze dla każdego modelu:*
- Random Forest: `x` (dokładność: `x`)
- Logistic Regression: `x` (dokładność: `x`)  
- Neural Network: `x` (dokładność: `x`)

= Wnioski i obserwacje

*Wpływ balansowania danych:*
- Przed balansowaniem: `x` opóźnionych, `x` nieopóźnionych lotów
- Po balansowaniu: `x` opóźnionych, `x` nieopóźnionych lotów
- Wpływ na dokładność: `x`

*Najważniejsze cechy wpływające na opóźnienia:*
1. `x`
2. `x`
3. `x`

*Optymalizacja genetyczna vs. domyślne parametry:*
- Średnia poprawa dokładności: `x`%
- Najlepsza poprawa: `x` dla modelu `x`

= Rekomendacje
1. *Model produkcyjny:* `x` z parametrami `x`
2. *Kluczowe cechy:* Skupić się na `x`, `x`, `x`
3. *Dalsze badania:* 
   - Testowanie na pełnym zbiorze danych
   - Dodanie cech pogodowych
   - Analiza sezonowości opóźnień

= Podsumowanie
Projekt wykazał skuteczność `x` w predykcji opóźnień lotów. Algorytm genetyczny pozwolił na `x` poprawę wyników względem parametrów domyślnych. Najważniejszymi czynnikami wpływającymi na opóźnienia okazały się `x`.

*Osiągnięte cele:*
- Dokładność najlepszego modelu: `x`%
- Identyfikacja kluczowych cech
- Optymalizacja hiperparametrów
- Analiza wpływu poszczególnych cech

*Kod źródłowy:*
- `model_comparison.py` - porównanie modeli z domyślnymi parametrami
- `genetic_tuning.py` - optymalizacja genetyczna
- `benchmark_analysis.py` - analiza wpływu cech



/*
Original delayed flights: 1023498
Original non-delayed flights: 4795581
Balanced delayed flights: 50000
Balanced non-delayed flights: 50000
Dataset shape: (100000, 13)
Target distribution:
DELAYED
1    50000
0    50000
Name: count, dtype: int64

Training Decision Tree...
Decision Tree Results:
Accuracy: 0.5484
              precision    recall  f1-score   support

           0      0.547     0.547     0.547      9965
           1      0.550     0.549     0.550     10035

    accuracy                          0.548     20000
   macro avg      0.548     0.548     0.548     20000
weighted avg      0.548     0.548     0.548     20000


Training Random Forest...
Random Forest Results:
Accuracy: 0.6157
              precision    recall  f1-score   support

           0      0.612     0.624     0.618      9965
           1      0.619     0.608     0.613     10035

    accuracy                          0.616     20000
   macro avg      0.616     0.616     0.616     20000
weighted avg      0.616     0.616     0.616     20000


Training Logistic Regression...
/mnt/archive/files/Studia/ED/ED/etap2/.venv/lib64/python3.13/site-packages/sklearn/linear_model/_logistic.py:470: ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=1000).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Logistic Regression Results:
Accuracy: 0.5907
              precision    recall  f1-score   support

           0      0.591     0.580     0.586      9965
           1      0.591     0.601     0.596     10035

    accuracy                          0.591     20000
   macro avg      0.591     0.591     0.591     20000
weighted avg      0.591     0.591     0.591     20000


Training KNN...
KNN Results:
Accuracy: 0.5530
              precision    recall  f1-score   support

           0      0.552     0.543     0.547      9965
           1      0.554     0.563     0.558     10035

    accuracy                          0.553     20000
   macro avg      0.553     0.553     0.553     20000
weighted avg      0.553     0.553     0.553     20000


Training Neural Network...
Neural Network Results:
Accuracy: 0.5932
              precision    recall  f1-score   support

           0      0.600     0.549     0.573      9965
           1      0.587     0.637     0.611     10035

    accuracy                          0.593     20000
   macro avg      0.594     0.593     0.592     20000
weighted avg      0.594     0.593     0.592     20000


Summary of Model Accuracies:
Decision Tree  : 0.5484
Random Forest  : 0.6157
Logistic Regression: 0.5907
KNN            : 0.5530
Neural Network : 0.5932

Results saved to model_comparison_results.json

=== FEATURE IMPORTANCE ANALYSIS ===
Using limited dataset: 10,000 rows
Feature Importance Ranking:
 1. FLIGHT_NUMBER       : 0.1349
 2. SCHEDULED_ARRIVAL   : 0.1249
 3. SCHEDULED_DEPARTURE : 0.1211
 4. TAIL_NUMBER         : 0.1208
 5. DESTINATION_AIRPORT : 0.1138
 6. DISTANCE            : 0.1084
 7. ORIGIN_AIRPORT      : 0.1044
 8. SCHEDULED_TIME      : 0.1017
 9. AIRLINE             : 0.0698
10. DAY_OF_WEEK         : 0.0000
11. MONTH               : 0.0000
12. YEAR                : 0.0000
13. DAY                 : 0.0000

=== PERFORMANCE SUMMARY ===

Baseline Performance (All Features):
RandomForest        : 0.6224
LogisticRegression  : 0.5861
NeuralNetwork       : 0.5423

Feature Removal Impact:
Scenario             RF       LR       NN       Features  
------------------------------------------------------------
No AIRLINE           0.6118   0.5801   0.5196   12        
No ORIGIN_AIRPORT    0.6299   0.5846   0.5227   12        
No DESTINATION_AIRPORT 0.6329   0.5801   0.5574   12        
No DAY_OF_WEEK       0.6148   0.5861   0.5302   12        
No MONTH             0.6148   0.5861   0.5302   12        
No DISTANCE          0.6224   0.5650   0.5332   12        
No SCHEDULED_DEPARTURE 0.6208   0.5710   0.5196   12        
No SCHEDULED_TIME    0.6239   0.5619   0.5151   12        
No Airport Info      0.6239   0.5846   0.5166   11        
No Time Info         0.6344   0.5619   0.5483   9         
Categories Only      0.5831   0.5030   0.5136   6         
Numerical Only       0.6027   0.5770   0.5287   8         
Essential Only       0.6163   0.5665   0.5650   9         

Best Performing Scenarios:
RandomForest: No Time Info (0.6344)
LogisticRegression: No DAY_OF_WEEK (0.5861)
NeuralNetwork: Essential Only (0.5650)

Results saved to benchmark_analysis_results.json

Benchmarking complete!

RandomForest Results:
Best Parameters: {'n_estimators': 165, 'max_depth': 17, 'min_samples_split': 14, 'min_samples_leaf': 2}
Best Data Size: 1000
Best CV Score: 0.6580

LogisticRegression Results:
Best Parameters: {'C': 34.0739604245577, 'max_iter': 912}
Best Data Size: 1000
Best CV Score: 0.6320

NeuralNetwork Results:
Best Parameters: {'hidden_layer_sizes': (135, 20), 'alpha': 0.007314074379494969, 'max_iter': 933}
Best Data Size: 1000
Best CV Score: 0.5520

Testing optimized models...
RandomForest: 0.6912
LogisticRegression: 0.6105
NeuralNetwork: 0.6254

*/