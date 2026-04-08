import pandas as pd
import numpy as np
import pickle


# ===============================
# Chargement du modèle XGBoost
# ===============================

with open("model_xgboost.pkl", "rb") as f:
    model_xgboost = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("history.pkl", "rb") as f:
    history = pickle.load(f)


# ===============================
# Fonctions utilitaires
# ===============================

def get_store_dept_history(store, dept):
    hist = history[(history["Store"] == store) & (history["Dept"] == dept)].copy()
    hist = hist.sort_values("Date")
    return hist


def compute_lags(store, dept):
    hist = get_store_dept_history(store, dept)

    if len(hist) < 4:
        raise ValueError("Pas assez d'historique pour ce Store/Dept")

    return (
        hist["Weekly_Sales"].iloc[-1],  # Lag_1
        hist["Weekly_Sales"].iloc[-2],  # Lag_2
        hist["Weekly_Sales"].iloc[-4],  # Lag_4
    )


def compute_rolling(store, dept):
    hist = get_store_dept_history(store, dept)

    if len(hist) < 4:
        raise ValueError("Pas assez d'historique pour ce Store/Dept")

    last_4 = hist["Weekly_Sales"].iloc[-4:]
    return last_4.mean(), last_4.std()


def compute_encodings(store, dept):
    key = f"{store}_{dept}"

    store_enc = encoders["store_mean"].get(store, encoders["global_mean"])
    dept_enc = encoders["dept_mean"].get(dept, encoders["global_mean"])
    store_dept_enc = encoders["store_dept_mean"].get(key, encoders["global_mean"])

    return store_enc, dept_enc, store_dept_enc


def compute_date_features(date_str):
    date = pd.to_datetime(date_str)
    week_of_year = int(date.isocalendar().week)

    week_sin = np.sin(2 * np.pi * week_of_year / 52)
    week_cos = np.cos(2 * np.pi * week_of_year / 52)

    return date, date.year, date.month, week_of_year, week_sin, week_cos


# ===============================
# Prédiction
# ===============================

def predict_sales(
    store,
    dept,
    date_str,
    temperature,
    fuel_price,
    cpi,
    unemployment,
    is_holiday,
    markdown1,
    markdown2,
    markdown3,
    markdown4,
    markdown5,
    size,
    type_store,
    black_friday,
    thanksgiving,
    xmas_week
):
    date, year, month, week_of_year, week_sin, week_cos = compute_date_features(date_str)

    store_enc, dept_enc, store_dept_enc = compute_encodings(store, dept)
    lag_1, lag_2, lag_4 = compute_lags(store, dept)
    rolling_mean_4, rolling_std_4 = compute_rolling(store, dept)

    type_a = int(type_store == "A")
    type_b = int(type_store == "B")
    type_c = int(type_store == "C")
    is_promo = int((markdown1 + markdown2 + markdown3 + markdown4 + markdown5) > 0)
    input_data = pd.DataFrame([{
        "Store": store,
        "Dept": dept,
        "IsHoliday": is_holiday,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Size": size,
        "MarkDown1": markdown1,
        "MarkDown2": markdown2,
        "MarkDown3": markdown3,
        "MarkDown4": markdown4,
        "MarkDown5": markdown5,
        "Is_Promo": is_promo,
        "Year": year,
        "Month": month,
        "WeekOfYear": week_of_year,
        "Week_sin": week_sin,
        "Week_cos": week_cos,
        "Store_enc": store_enc,
        "Dept_enc": dept_enc,
        "Store_Dept_enc": store_dept_enc,
        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_4": lag_4,
        "Rolling_Mean_4": rolling_mean_4,
        "Rolling_Std_4": rolling_std_4,
        "Black_Friday": black_friday,
        "Thanksgiving": thanksgiving,
        "Xmas_Week": xmas_week,
        "Type_A": type_a,
        "Type_B": type_b,
        "Type_C": type_c
        
    }])
  
    # Réordonner selon les features du modèle entraîné
    input_data = input_data[features]

    prediction_log = model_xgboost.predict(input_data)[0]
    prediction = np.exp(prediction_log)

    return prediction