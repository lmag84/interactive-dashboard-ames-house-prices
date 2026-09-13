import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np


# Utworzenie strony w Streamlit
st.set_page_config(layout="wide", page_title="Ceny nieruchomości w mieście Ames w USA w latach 1872-2010")

# Wczytanie pliku oraz wyfiltrowanie danych do projektu
@st.cache_data
def load_data():

    df = pd.read_csv("AmesHousing.csv")
    columns = [
        "SalePrice",
        "Gr Liv Area",
        "Lot Area",
        "Full Bath",
        "Bedroom AbvGr",
        "Overall Qual",
        "Year Built",   ]
    df=df.dropna(subset=columns)
    df=df[columns]

    df = df.rename(columns={
        "Gr Liv Area": "Powierzchnia domu",
        "Lot Area": "Powierzchnia działki",
        "Overall Qual": "Jakość",
        "Year Built": "Rok budowy",
        "Full Bath": "Ilość łazienek",
        "Bedroom AbvGr": "Ilość pokoi",
        "SalePrice": "Cena"
    })

    return df

df=load_data()


# Utworzenie menu aplikacji
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Przejdź do:", ["Wprowadzenie", "Eksploracja danych", "Model"])

# Przetrenowanie danych na modelach, porównanie wyników
def train_models(X_train, X_test, y_train, y_test):

    trained_models = {}
    metrics = {"model": [], "r2": [], "mae": [], "rmse": []}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(eval_metric="rmse", random_state=42)
    }

    for model_name, model in models.items():

        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()

        if model_name == "Linear Regression":
            scaler = StandardScaler()

            X_train_copy = pd.DataFrame(
                scaler.fit_transform(X_train_copy),
                columns=X_train_copy.columns,
                index=X_train_copy.index
            )

            X_test_copy = pd.DataFrame(
                scaler.transform(X_test_copy),
                columns=X_test_copy.columns,
                index=X_test_copy.index
            )

            model.fit(X_train_copy, y_train)
            y_pred = model.predict(X_test_copy)

            trained_models[model_name] = {
                "model": model,
                "scaler": scaler
            }

        else:
            model.fit(X_train_copy, y_train)
            y_pred = model.predict(X_test_copy)

            trained_models[model_name] = {
                "model": model,
                "scaler": None
            }

        r2, mae, rmse = evaluate_model(y_test, y_pred)

        metrics["model"].append(model_name)
        metrics["r2"].append(r2)
        metrics["mae"].append(mae)
        metrics["rmse"].append(rmse)

    results_df = pd.DataFrame(metrics)

    return trained_models, results_df


# Funkcja oceny modeli
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

# Filtrowanie danych
def filtered_data(df):
    st.sidebar.header("Filtry danych")
    pow_budynku_min, pow_budynku_max = st.sidebar.slider("Powierzchnia domu (sq feet)",
                                                         int(df["Powierzchnia domu"].min()),
                                                         int(df["Powierzchnia domu"].max()), (1000, 3000))
    pow_dzialki_min, pow_dzialki_max = st.sidebar.slider("Powierzchnia działki (sq feet)",
                                                         int(df["Powierzchnia działki"].min()),
                                                         int(df["Powierzchnia działki"].max()), (6000, 16000))
    jakosc = st.sidebar.multiselect("Jakośc (1-10)", sorted(df["Jakość"].unique()), default=[6, 7, 8])
    rok_budowy_min, rok_budowy_max = st.sidebar.slider("Rok budowy (1872-2010)", int(df["Rok budowy"].min()),
                                                       int(df["Rok budowy"].max()), (1980, 1990))
    filtered_df = df[
        (df["Powierzchnia domu"].between(pow_budynku_min, pow_budynku_max)) &
        (df["Powierzchnia działki"].between(pow_dzialki_min, pow_dzialki_max)) &
        (df["Jakość"].isin(jakosc)) &
        (df["Rok budowy"].between(rok_budowy_min, rok_budowy_max))
        ]

    return filtered_df


# Tworzenie podstrony Wprowadzenie
if page =="Wprowadzenie":
    st.title(":blue[Wprowadzenie]", text_alignment="center")
    st.markdown("""
    Dashboard prezentuje dane z sprzedaży domów w latach 1872-2010 w mieście Aims w USA. 
    
    Na podstronie eksploracja danych zaprezentowane są dane pobrane ze strony internetowej Kaggle. Dane można filtrowac w oparciu o wybrane cechy.
    W tej sekcji zawarte są również:
    - wykresy zależności wielkości domu i działki na której stoi dom od ceny, 
    - wykres pudełkowy zależności jakości domu od jego ceny,
    - korelacja między wybranymi cechami domów,
    - podział danych na klastry.
    
    Na podstronie model można zasymulowac cenę domu o wybranych parametrach.
    Symulacja jest przeprowadzona w oparciu o wytrenowanie modelu Random Forest.
    

    ---
    Dane pochodzą z: [House Prices - Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data) 
       
    ---
    
    Dashboard został stworzony w języku Python z wykorzystaniem biblioteki Streamlit.
    
    Plik zawiera listę bibliotek potrzebnych do uruchomienia aplikacji.
    """)

    # wstawienie przycisku, który umożliwia pobranie pliku z wymaganiami programowymi do projektu
    st.download_button(
        label="Pobierz wymagane biblioteki (requirements.txt)",
        data="""streamlit==1.45.1
pandas==1.5.3
numpy==1.23.5
plotly==6.1.2
scikit-learn==1.2.2
statsmodels==0.14.4""",
file_name="requirements.txt"
    )

# Tworzenie podstrony Eksploracja danych
elif page =="Eksploracja danych":

    st.title(":blue[Eksploracja danych]", text_alignment="center")

    st.info("Użyj dostępnych filtrów, aby sprawdzić jak zmieniają się ceny domów w zależności od ich roku budowy, jakości, powierzchni oraz powierzchni działki na której są zbudowane.")

    # Stworzenie filtrów danych
    filtered_df = filtered_data(df)

    st.success(f"**Liczba domów spełniająca kryteria: {len(filtered_df)}**")

    sort_option = st.selectbox("Wyfiltrowane wyniki sortuj po:", filtered_df.columns)

    if st.checkbox("Pokaż wyfiltrowane dane w formie tabelki"):
        st.dataframe(filtered_df.sort_values(by=sort_option))

    st.divider()

    # Wykres z domami z wyfiltrowanych danych
    if st.checkbox("Pokaż wyfiltrowane dane na wykresie."):
        st.subheader(":red[Rozkład cen po wyfiltrowanych danych]",text_alignment="center")
        fig = px.histogram(filtered_df, x="Cena", nbins=40)
        fig.update_layout(
            xaxis_title = "Cena domu (USD)",
            yaxis_title = "Liczba domów spełniająca kryteria"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # Wykres zależności powierzchni domu i działki od ceny

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(":red[Cena domu vs powierzchnia domu]",text_alignment="center")
        fig1 = px.scatter(df, x="Powierzchnia domu", y="Cena", labels={"Powierzchnia domu":"Powierzchnia domu (sq feet)", "Cena":"Cena (USD)"}, trendline="ols")
        st.plotly_chart(fig1)

    with col2:
        st.subheader(":red[Cena domu vs powierzchnia działki]", text_alignment="center")
        fig2 = px.scatter(df, x="Powierzchnia działki", y="Cena", labels={"Powierzchnia działki": "Powierzchnia działki (sq feet)", "Cena": "Cena (USD)"}, trendline="ols")
        st.plotly_chart(fig2)

    st.info("""
    Wnioski z analizy powyższych wykresów:
    - Widać wyraźną zależność między powierzchnią domu a jego ceną — większe domy są zazwyczaj droższe.
    - Dla domów o powierzchni powyżej ~4000 sq feet zależność ta jest mniej wyraźna, co sugeruje wpływ innych czynników na cenę domu.
    - Nie obserwuje się silnej zależności między powierzchnią działki a ceną domu.
    - Najdroższe domy niekoniecznie mają największe działki — często mają one przeciętną wielkość.
    """)

    st.divider()

    # Wykres pudełkowy zależności jakości od ceny
    st.subheader(":red[Cena domu vs. jakość domu]",text_alignment="center")
    st.plotly_chart(px.box(df, x="Jakość", y="Cena", labels={"Jakość":"Jakość", "Cena":"Cena domu (USD)"}))

    st.info("""
    Wnioski z analizy poyższego wykresu:
    - Widać wyraźną zależność między jakością domu a jego ceną — wysokiej jakości domy są zazwyczaj droższe.
    - Największa rozpiętość cenowa jest w domach o najwyższej jakości, co sugeruje większe zróżnicowanie w tej grupie.
    - Mediana ceny rośnie wraz ze wzrostem jakości, co wskazuje na wpływ tej cechy na wartość nieruchomości.
    - Dla niższych poziomów jakości ceny są bardziej skupione i mniej zróżnicowane.
    """)

    st.divider()

    # wykres heatmap korelacji
    st.subheader(":red[Korelacja między wybranymi cechami domu]", text_alignment="center")
    corr = df.corr(numeric_only=True)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.columns),
        annotation_text=corr.round(2).values,
        colorscale="Greys"
    )
    st.plotly_chart(fig)

    st.info("""
    Wnioski z analizy poyższej tabeli:
    - Najsilniejsza zależnośc występuje między ceną a jakością (r=0.8), co wskazuje na duży wpływ jakości domu na jego cenę.
    - Również wysoką korelacją związana jest cena oraz powierzchnia domu (r=0.71), co oznacza że większe domy są zazwyczaj droższe.
    - Mały wpływ na cenę domu ma ilość zawartych w nim pokoi (r=0.14)  oraz powierzchnia działki (r=0.27) na której się znajduje dom.
    - Jakość domu nie zależy istotnie od powierzchni działki (r = 0.1), co może sugerować, że standard wykończenia jest niezależny od wielkości posesji.
    """)

    st.divider()


    # Podział danych na 3 klastry za pomocą KMeans
    st.subheader(":red[Podział danych na klastry]", text_alignment="center")
    df_clusters = df.copy()

    features = df_clusters[[
        "Powierzchnia domu",
        "Powierzchnia działki",
        "Jakość",
        "Rok budowy",
        "Cena"
    ]]
    # Przeskalowanie danych
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)

    df_clusters["Clusters"] = kmeans.fit_predict(scaled_features)

    st.plotly_chart(px.scatter(df_clusters, x="Powierzchnia domu", y="Cena", color="Clusters"))

    if st.checkbox("Pokaż średnie wartości dla każdego klastra"):
        clusters_mean = round(df_clusters.groupby("Clusters").mean(), 1)
        st.dataframe(clusters_mean)

    st.info("""
    Wnioski z analizy powyższego wykresu:
    - Segment budżetowy (Cluster nr 2) - to najtańsze domy – nieruchomości najstarsze, najmniejsze oraz o najmniejszej liczbie łazienek i niższej jakości.
    - Segment premium (Cluster nr 1) - to najdroższe domy – charakteryzują się one najwyższą jakością, największą powierzchnią użytkową oraz największymi działkami.
    - Segment średni (Cluster nr 0) - to domy o średnich wartościach wszystkich analizowanych cech.
    - Największy wpływ na podział na klastry mają jakość domu oraz jego powierzchnia.
    """)


# Tworzenie podstrony Model
elif page =="Model":
    st.title(":red[Predykcja cen domów]", text_alignment="center")
    st.markdown("""
    Na podstawie poniższych zmiennych z danych dotyczących sprzedaży domów w latach 1872-2010 w mieście Ames w USA zostały zbudowane modele do predykcji cen domów:
    - powierzchnia budynku,
    - powierzchnia działki,
    - jakośc,
    - rok budowy,
    - ilośc łazienek,
    - ilośc pokoi.
    
    Poniżej przedstawiona tabelka prezentuje współczynnik determinacji (R2), średni błąd bezwzględny (mae) oraz pierwiastek średniego błędu kwadratowego (RMSE) 
    dla wybranych metod statystycznych: Liear Regression, Random Forest, XGBoost.
    """)
    X = df.drop(columns="Cena")
    y = df["Cena"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_models, results_df = train_models(X_train, X_test, y_train, y_test)

    best_model = trained_models["Random Forest"]

    st.dataframe(results_df)
    st.success(f"""
    Model regresji liniowej posiada dobre wyniki (R2=0.78), ale gorsze niż Random Forest i XGboost.
    Najlepszym modelem  w powyższym przypadku jest **Random Forest**, który osiągnął najwyższe R2 oraz najniższe wartości błędów MAE i RMSE.
    W związku z powyższym do predykcji cen użyję modelu Random Forest """)

    # Stworzenie formularza do peredykcji ceny domu o wybranych parametrach
    with st.form("Formularz predykcji:"):
        st.subheader("Wprowadź dane domu, a następnie kliknij na przycisk Oblicz cenę")

        powierzchnia_dzialki_input = st.number_input("Powierzchnia działki (sq feet)", min_value=1300, max_value=21000, step=1000)
        powierzchnia_domu_input = st.number_input("Powierzchnia domu (sq feet)", min_value=350, max_value= 5600, step=100)
        jakosc_input = st.number_input("Jakość domu", min_value = int(df["Jakość"].min()), max_value=int(df["Jakość"].max()))
        rok_budowy_input = st.number_input("Rok budowy domu", min_value=1890, max_value=2010, step=1)
        ilosc_lazienek_input = st.number_input("Ilość łazienek", min_value=int(df["Ilość łazienek"].min()), max_value=int(df["Ilość łazienek"].max()), step=1)
        ilosc_pokoi_input = st.number_input("Ilość pokoi", min_value=int(df["Ilość pokoi"].min()), max_value=int(df["Ilość pokoi"].max()), step=1)

        submit = st.form_submit_button(":red[Oblicz cenę]", type="secondary")

    if submit:
        input_data = pd.DataFrame([[
            powierzchnia_domu_input,
            powierzchnia_dzialki_input,
            ilosc_lazienek_input,
            ilosc_pokoi_input,
            jakosc_input,
            rok_budowy_input
        ]], columns=[
            "Powierzchnia domu",
            "Powierzchnia działki",
            "Ilość łazienek",
            "Ilość pokoi",
            "Jakość",
            "Rok budowy"
        ])

        model = trained_models["Random Forest"]["model"]

        pred_price = model.predict(input_data)[0]

        st.success(f"Prognozowana cena domu {pred_price:,.0f} USD")


