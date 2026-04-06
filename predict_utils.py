import pandas as pd
import numpy as np
import pickle
import os
import gdown

def download_if_missing(file_id, output_name):
    if not os.path.exists(output_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)

# ===============================
# Chargement des modèles et fichiers
# ===============================

RF_FILE_ID = "1KbS_2Q6L4ERXC4Pz1-ALQX4vsTW6nyAu"
XGB_FILE_ID = "17Lz_bBLa6y5JbK9mjKd83Uql1HT55XKk"

download_if_missing(RF_FILE_ID, "model_rf.pkl")
download_if_missing(XGB_FILE_ID, "model_xgb.pkl")

with open("model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open("model_xgb.pkl", "rb") as f:
    model_xgb = pickle.load(f)
with open("features_final.pkl", "rb") as f:
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

    lag_1 = hist["Weekly_Sales"].iloc[-1]
    lag_2 = hist["Weekly_Sales"].iloc[-2]
    lag_4 = hist["Weekly_Sales"].iloc[-4]

    return lag_1, lag_2, lag_4


def compute_rolling(store, dept):
    hist = get_store_dept_history(store, dept)

    if len(hist) < 4:
        raise ValueError("Pas assez d'historique pour ce Store/Dept")

    last_4 = hist["Weekly_Sales"].iloc[-4:]

    rolling_mean_4 = last_4.mean()
    rolling_std_4 = last_4.std()

    return rolling_mean_4, rolling_std_4


def compute_encodings(store, dept):
    store_dept_key = f"{store}_{dept}"

    store_enc = encoders["store_mean"].get(store, encoders["global_mean"])
    dept_enc = encoders["dept_mean"].get(dept, encoders["global_mean"])
    store_dept_enc = encoders["store_dept_mean"].get(store_dept_key, encoders["global_mean"])

    return store_enc, dept_enc, store_dept_enc


def compute_date_features(date_str):
    date = pd.to_datetime(date_str)

    year = date.year
    month = date.month
    week = int(date.isocalendar().week)

    return date, year, month, week


# ===============================
# Prédiction 1 semaine
# ===============================

def predict_sales(store, dept, date_str, is_holiday,
                  temperature, fuel_price,
                  markdown1, markdown2, markdown3, markdown4, markdown5,
                  cpi, unemployment, size,
                  type_store,
                  black_friday, thanksgiving, xmas_week,
                  model_type="rf"):

    date, year, month, week = compute_date_features(date_str)

    # encodages
    store_enc, dept_enc, store_dept_enc = compute_encodings(store, dept)

    # lags
    lag_1, lag_2, lag_4 = compute_lags(store, dept)

    # rolling
    rolling_mean_4, rolling_std_4 = compute_rolling(store, dept)

    # promo
    is_promo = int(
        (markdown1 > 0) or
        (markdown2 > 0) or
        (markdown3 > 0) or
        (markdown4 > 0) or
        (markdown5 > 0)
    )

    # type de store
    type_a = int(type_store == "A")
    type_b = int(type_store == "B")
    type_c = int(type_store == "C")

    # construire la ligne complète
    input_data = pd.DataFrame([{
        "Store": store,
        "Dept": dept,
        "IsHoliday": is_holiday,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "MarkDown1": markdown1,
        "MarkDown2": markdown2,
        "MarkDown3": markdown3,
        "MarkDown4": markdown4,
        "MarkDown5": markdown5,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Size": size,
        "Year": year,
        "Month": month,
        "Week": week,
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
        "Is_Promo": is_promo,
        "Type_A": type_a,
        "Type_B": type_b,
        "Type_C": type_c
    }])

    # remettre les colonnes dans le bon ordre
    input_data = input_data[features]

    # prédiction sur la cible log
    if model_type == "xgb":
        prediction_log = model_xgb.predict(input_data)[0]
    else:
        prediction_log = model_rf.predict(input_data)[0]

    # retour en valeur réelle
    prediction_real = np.exp(prediction_log)

    return prediction_real


# ===============================
# Test local
# ===============================

if __name__ == "__main__":
    pred = predict_sales(
        store=1,
        dept=1,
        date_str="2012-10-26",
        is_holiday=0,
        temperature=70.0,
        fuel_price=3.5,
        markdown1=0.0,
        markdown2=0.0,
        markdown3=0.0,
        markdown4=0.0,
        markdown5=0.0,
        cpi=220.0,
        unemployment=7.0,
        size=151315,
        type_store="A",
        black_friday=0,
        thanksgiving=0,
        xmas_week=0,
        model_type="rf"
    )

    print("Prédiction :", pred)