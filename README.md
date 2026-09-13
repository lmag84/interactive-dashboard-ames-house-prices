# Analiza i predykcja cen domów w Ames (USA)

Interaktywna aplikacja webowa napisana w języku **Python** przy użyciu biblioteki **Streamlit**, przeznaczona do eksploracji danych o nieruchomościach oraz estymacji cen domów w mieście **Ames (USA)** na podstawie danych z lat 1872–2010.

---

##  Spis treści
- [O projekcie](#o-projekcie)
- [Struktura danych](#struktura-danych)
- [Główne funkcjonalności](#główne-funkcjonalności)
- [Użyte technologie](#użyte-technologie)
- [Jak uruchomić projekt lokalnie?](#jak-uruchomić-projekt-lokalnie)
- [Modele uczenia maszynowego](#modele-uczenia-maszynowego)

---

##  O projekcie

Projekt wykorzystuje zestaw danych **Ames Housing Dataset** pobrany z serwisu Kaggle. Głównym celem aplikacji jest dostarczenie interaktywnego narzędzia analitycznego dla rynku nieruchomości oraz umożliwienie użytkownikowi przewidywania szacunkowej wartości domu na podstawie wprowadzonych parametrów.

Aplikacja składa się z trzech głównych modułów (podstron):
1. **Wprowadzenie:** Omówienie zakresu analizy oraz możliwość pobrania pliku z wymaganiami.
2. **Eksploracja danych:** Filtrowanie danych, wizualizacje korelacji, wykresy zależności oraz segmentacja klientów (klasteryzacja KMeans).
3. **Model:** Porównanie wydajności modeli regresyjnych oraz interaktywny formularz do predykcji cen.

---

## Struktura danych

Aplikacja bazuje na pliku `AmesHousing.csv` i analizuje kluczowe cechy nieruchomości:

* `SalePrice` (Cena) – cena sprzedaży nieruchomości w USD
* `Gr Liv Area` (Powierzchnia domu) – powierzchnia mieszkalna w stopach kwadratowych (`sq feet`)
* `Lot Area` (Powierzchnia działki) – powierzchnia posesji w stopach kwadratowych (`sq feet`)
* `Overall Qual` (Jakość) – ogólna jakość wykończenia i materiałów (skala 1–10)
* `Year Built` (Rok budowy) – rok wzniesienia budynku
* `Full Bath` (Ilość łazienek) – liczba pełnych łazienek
* `Bedroom AbvGr` (Ilość pokoi) – liczba sypialni powyżej poziomu gruntu

---

## Główne funkcjonalności

* **Interaktywne filtrowanie danych:** Dynamiczne dopasowywanie zakresu powierzchni, jakości oraz roku budowy nieruchomości za pomocą suwaków w panelu bocznym.
* **Zaawansowana wizualizacja (Plotly):**
  * Histogramy rozkładu cen.
  * Wykresy punktowe ze ścieżką trendu OLS.
  * Wykresy pudełkowe (Box-plot) zależności ceny od jakości.
  * Macierz korelacji cech w formie interaktywnej mapy ciepła (Heatmap).
* **Klasteryzacja KMeans:** Segmentacja rynku na 3 grupy (Segment budżetowy, średni oraz premium) na podstawie standaryzowanych cech.
* **Interaktywny kalkulator wyceny:** Formularz pozwalający wygenerować prognozę ceny domu w czasie rzeczywistym.

---

## Użyte technologie

* **Python** (3.10+)
* **Streamlit** – interfejs graficzny użytkownika
* **Pandas & NumPy** – manipulacja i analiza danych
* **Plotly Express & Figure Factory** – interaktywna wizualizacja danych
* **Scikit-Learn** – budowa modeli regresji, przeskalowywanie (StandardScaler) i klasteryzacja (KMeans)
* **XGBoost** – algorytm uśredniania drzew decyzyjnych

---

##  Jak uruchomić projekt lokalnie?

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/lmag84/ames-housing-dashboard.git
cd ames-housing-dashboard
```
### 2. Przygotowanie danych 
Upewnij się, że plik AmesHousing.csv znajduje się w głównym katalogu projektu.
### 3. Instalacja zależności
Zainstaluj wymagane pakiety za pomocą pip:
```bash
pip install -r requirements.txt
```

Zawartość pliku requirements.txt:
````
streamlit==1.45.1
pandas==1.5.3
numpy==1.23.5
plotly==6.1.2
scikit-learn==1.2.2
statsmodels==0.14.4
xgboost
````
### 4. Uruchomienie aplikacji
Uruchom serwer aplikacji Streamlit:
````
streamlit run dashboard.py
````
## Modele uczenia maszynowego
W aplikacji porównywane są trzy algorytmy regresyjne ewaluowane na podstawie wskaźników $R^2$, MAE oraz RMSE:Regresja Liniowa (z automatycznym przeskalowaniem danych za pomocą StandardScaler).Random Forest Regressor (100 drzew estymacyjnych).XGBoost Regressor.Do końcowej predykcji cen w formularzu wykorzystywany jest model Random Forest, jako charakteryzujący się najwyższym współczynnikiem determinacji oraz najniższym poziomem błędów w analizowanym zbiorze.