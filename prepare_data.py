import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


# =========================================================
# A - Chargement et fusion des données
# =========================================================

def load_data():
    train = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")
    stores = pd.read_csv("data/stores.csv")

    print("Train shape:", train.shape)
    print("Features shape:", features.shape)
    print("Stores shape:", stores.shape)

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    print("Merged shape:", df.shape)
    print(df.columns.tolist())
    print()
    print(df.dtypes)

    return df


# =========================================================
# B1 - Préparation initiale
# =========================================================

def prepare_initial_data(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    print(df["Date"].dtype)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    print(df[["Date", "Year", "Month", "Week"]].head())
    print(df.isnull().sum())

    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print("MarkDown remplis")

    # Suppression des ventes non positives avant log
    df = df[df["Weekly_Sales"] > 0]
    print("Après suppression des ventes négatives :", df.shape)

    # Transformation logarithmique de la cible
    df["Weekly_Sales_Log"] = np.log(df["Weekly_Sales"])
    print(df[["Weekly_Sales", "Weekly_Sales_Log"]].head())

    # Clé Store + Dept
    df["Store_Dept"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    print(df[["Store", "Dept", "Store_Dept"]].head())

    # Encodages globaux initiaux
    store_mean = df.groupby("Store")["Weekly_Sales_Log"].mean()
    df["Store_enc"] = df["Store"].map(store_mean)
    print(df[["Store", "Store_enc"]].head())

    dept_mean = df.groupby("Dept")["Weekly_Sales_Log"].mean()
    df["Dept_enc"] = df["Dept"].map(dept_mean)
    print(df[["Dept", "Dept_enc"]].head())

    store_dept_mean = df.groupby("Store_Dept")["Weekly_Sales_Log"].mean()
    df["Store_Dept_enc"] = df["Store_Dept"].map(store_dept_mean)
    print(df[["Store_Dept", "Store_Dept_enc"]].head())

    return df


# =========================================================
# B2 - Création des lags et rolling features
# =========================================================

def create_lags_and_rolling(df):
    df = df.copy()

    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    df["Lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
    df["Lag_2"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(2)
    df["Lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

    df["Rolling_Mean_4"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .transform(lambda x: x.shift(1).rolling(window=4).mean())
    )

    df["Rolling_Std_4"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .transform(lambda x: x.shift(1).rolling(window=4).std())
    )

    avant = len(df)
    df = df.dropna()
    apres = len(df)

    print(f"Lignes avant suppression : {avant:,}")
    print(f"Lignes supprimées        : {avant-apres:,} soit {(avant-apres)/avant*100:.2f}%")
    print(f"Lignes conservées        : {apres:,} soit {apres/avant*100:.2f}%")
    print(f"\nColonnes créées : {[col for col in df.columns if 'Lag' in col or 'Rolling' in col]}")

    return df


# =========================================================
# B3 - Identification des dates clés
# =========================================================

def identify_key_dates(df):
    thanksgiving = df[df["Date"].isin(pd.to_datetime([
        "2010-11-26", "2011-11-25"
    ]))][["Date"]].drop_duplicates()

    black_friday = pd.to_datetime(["2010-12-03", "2011-12-02"])
    xmas_week = pd.to_datetime(["2010-12-24", "2011-12-23"])

    print("Thanksgiving :")
    print(thanksgiving.to_string(index=False))
    print()
    print("Black Friday :")
    for d in black_friday:
        print(f"  {d.date()}")
    print()
    print("Xmas Week :")
    for d in xmas_week:
        print(f"  {d.date()}")

    return thanksgiving, black_friday, xmas_week


# =========================================================
# B3 - Encodage des fêtes
# =========================================================

def encode_holidays(df):
    df = df.copy()

    black_friday_dates = pd.to_datetime(["2010-12-03", "2011-12-02"])
    df["Black_Friday"] = df["Date"].isin(black_friday_dates).astype(int)

    thanksgiving_dates = pd.to_datetime(["2010-11-26", "2011-11-25"])
    df["Thanksgiving"] = df["Date"].isin(thanksgiving_dates).astype(int)

    xmas_dates = pd.to_datetime(["2010-12-24", "2011-12-23"])
    df["Xmas_Week"] = df["Date"].isin(xmas_dates).astype(int)

    def encode_holiday(row):
        if row["Date"] in black_friday_dates:
            return "Black_Friday"
        if row["Date"] in xmas_dates:
            return "Xmas_Week"
        if not row["IsHoliday"]:
            return "None"
        month = row["Date"].month
        if month == 2:
            return "SuperBowl"
        if month == 9:
            return "LaborDay"
        if month == 11:
            return "Thanksgiving"
        if month == 12:
            return "NewYear"
        return "Other"

    df["Holiday_Type"] = df.apply(encode_holiday, axis=1)

    print(
        df[["Date", "IsHoliday", "Thanksgiving", "Black_Friday", "Xmas_Week", "Holiday_Type"]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .query("Holiday_Type != 'None'")
        .to_string(index=False)
    )

    return df


# =========================================================
# B3 - Conversion IsHoliday en numérique + suppression Holiday_Type
# =========================================================

def finalize_holiday_columns(df):
    df = df.copy()

    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = df.drop(columns=["Holiday_Type"])

    print(df["IsHoliday"].value_counts())
    print(f"\nColonnes restantes : {df.shape[1]}")
    print(df.columns.tolist())

    return df


# =========================================================
# B4 - Création de la variable Is_Promo
# =========================================================

def create_is_promo(df):
    df = df.copy()

    df["Is_Promo"] = (
        (df["MarkDown1"] > 0) |
        (df["MarkDown2"] > 0) |
        (df["MarkDown3"] > 0) |
        (df["MarkDown4"] > 0) |
        (df["MarkDown5"] > 0)
    ).astype(int)

    print(df["Is_Promo"].value_counts())
    print(f"\n% semaines avec promotion : {df['Is_Promo'].mean()*100:.2f}%")

    return df


# =========================================================
# B5 - One-Hot Encoding de la variable Type
# =========================================================

def encode_type(df):
    df = df.copy()

    dummies = pd.get_dummies(df["Type"], prefix="Type", drop_first=False).astype(int)
    df = pd.concat([df.drop(columns=["Type"]), dummies], axis=1)

    print(df[["Type_A", "Type_B", "Type_C"]].value_counts())
    print(f"\nDimensions : {df.shape}")
    print(df.columns.tolist())

    return df


# =========================================================
# C3 - Fonction WMAE personnalisée
# =========================================================

def wmae(y_true, y_pred, thanksgiving, black_friday, xmas_week):
    weights = pd.Series(1.0, index=y_true.index)
    weights[(thanksgiving == 1) | (black_friday == 1) | (xmas_week == 1)] = 5
    return round((weights * (y_true - y_pred).abs()).sum() / weights.sum(), 2)


# =========================================================
# C4 - Définition des features, cible et validation
# =========================================================

def build_model_data(df):
    df = df.copy()

    df = df.sort_values("Date").reset_index(drop=True)

    features = [col for col in df.columns if col not in ["Date", "Weekly_Sales", "Weekly_Sales_Log", "Store_Dept"]]
    target = "Weekly_Sales_Log"

    X = df[features]
    y = df[target]
    y_real = df["Weekly_Sales"]
    thanksgiving = df["Thanksgiving"]
    black_friday = df["Black_Friday"]
    xmas_week = df["Xmas_Week"]

    tscv = TimeSeriesSplit(n_splits=5)

    print(f"Nombre de features : {len(features)}")
    print(f"Lignes             : {len(X):,}")
    print(f"Cible              : {target}")
    print(f"Features           : {features}")

    folds = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        folds.append((train_idx, val_idx))

        train_dates = df.iloc[train_idx]["Date"]
        val_dates = df.iloc[val_idx]["Date"]

        print(f"Fold {fold}")
        print(f"  Train : {train_dates.min().date()} -> {train_dates.max().date()} ({len(train_idx)} lignes)")
        print(f"  Valid : {val_dates.min().date()} -> {val_dates.max().date()} ({len(val_idx)} lignes)")
        print()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X.dtypes)

    return df, X, y, y_real, thanksgiving, black_friday, xmas_week, tscv, folds


# =========================================================
# C5 - Target Encoding — fonction réutilisable
# =========================================================

def target_encoding(X_train, X_val, y_train):

    X_train["Store_Dept_key"] = X_train["Store"].astype(str) + "_" + X_train["Dept"].astype(str)
    X_val["Store_Dept_key"] = X_val["Store"].astype(str) + "_" + X_val["Dept"].astype(str)

    store_mean = y_train.groupby(X_train["Store"].values).mean()
    dept_mean = y_train.groupby(X_train["Dept"].values).mean()
    store_dept_mean = y_train.groupby(X_train["Store_Dept_key"].values).mean()

    X_train["Store_enc"] = X_train["Store"].map(store_mean)
    X_train["Dept_enc"] = X_train["Dept"].map(dept_mean)
    X_train["Store_Dept_enc"] = X_train["Store_Dept_key"].map(store_dept_mean)

    X_val["Store_enc"] = X_val["Store"].map(store_mean)
    X_val["Dept_enc"] = X_val["Dept"].map(dept_mean)
    X_val["Store_Dept_enc"] = X_val["Store_Dept_key"].map(store_dept_mean)

    global_mean = y_train.mean()
    X_val["Store_enc"] = X_val["Store_enc"].fillna(global_mean)
    X_val["Dept_enc"] = X_val["Dept_enc"].fillna(global_mean)
    X_val["Store_Dept_enc"] = X_val["Store_Dept_enc"].fillna(global_mean)

    X_train = X_train.drop(columns=["Store_Dept_key"])
    X_val = X_val.drop(columns=["Store_Dept_key"])

    return X_train, X_val


# =========================================================
# Pipeline complet de préparation
# =========================================================

def prepare_all_data():
    df = load_data()
    df = prepare_initial_data(df)
    df = create_lags_and_rolling(df)
    identify_key_dates(df)
    df = encode_holidays(df)
    df = finalize_holiday_columns(df)
    df = create_is_promo(df)
    df = encode_type(df)

    return build_model_data(df)