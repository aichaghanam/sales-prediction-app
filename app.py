from flask import Flask, render_template, request
from predict_utils import predict_sales, history
import pandas as pd

app = Flask(__name__)

# Chargement des infos fixes des magasins
stores_df = pd.read_csv("data/stores.csv")


def get_store_info(store_id):
    row = stores_df[stores_df["Store"] == store_id]

    if row.empty:
        return None, None

    size = int(row.iloc[0]["Size"])
    type_store = str(row.iloc[0]["Type"])

    return size, type_store


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # ===============================
            # 1) MODE EXCEL
            # ===============================
            file = request.files.get("file")

            if file and file.filename and file.filename.strip() != "":
                if not file.filename.lower().endswith(".xlsx"):
                    return render_template(
                        "index.html",
                        error="Format invalide (.xlsx requis)",
                        form_data={},
                        chart_labels=[],
                        chart_values=[]
                    )

                df = pd.read_excel(file)

                # Normalisation IsHoliday
                if "IsHoliday" in df.columns and df["IsHoliday"].dtype == object:
                    df["IsHoliday"] = (
                        df["IsHoliday"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .map({
                            "FAUX": 0,
                            "FALSE": 0,
                            "0": 0,
                            "NON": 0,
                            "VRAI": 1,
                            "TRUE": 1,
                            "1": 1,
                            "OUI": 1,
                        })
                    )

                # Colonnes événements par défaut si absentes
                for col in ["Black_Friday", "Thanksgiving", "Xmas_Week"]:
                    if col not in df.columns:
                        df[col] = 0

                predictions = []

                for _, row in df.iterrows():
                    store = int(row["Store"])
                    dept = int(row["Dept"])
                    date_str = str(row["Date"])

                    # Récupération automatique des vraies infos du store
                    size, type_store = get_store_info(store)
                    if size is None or type_store is None:
                        raise ValueError(f"Store introuvable dans stores.csv : {store}")

                    pred = float(predict_sales(
                        store=store,
                        dept=dept,
                        date_str=date_str,
                        temperature=float(row["Temperature"]),
                        fuel_price=float(row["Fuel_Price"]),
                        cpi=float(row["CPI"]),
                        unemployment=float(row["Unemployment"]),
                        is_holiday=int(row["IsHoliday"]),
                        markdown1=float(row["MarkDown1"]),
                        markdown2=float(row["MarkDown2"]),
                        markdown3=float(row["MarkDown3"]),
                        markdown4=float(row["MarkDown4"]),
                        markdown5=float(row["MarkDown5"]),
                        size=size,
                        type_store=type_store,
                        black_friday=int(row.get("Black_Friday", 0)),
                        thanksgiving=int(row.get("Thanksgiving", 0)),
                        xmas_week=int(row.get("Xmas_Week", 0)),
                    ))
                    predictions.append(pred)

                df["Prediction"] = predictions

                # Affichage de la première ligne dans l'interface
                first_row = df.iloc[0]

                store = int(first_row["Store"])
                dept = int(first_row["Dept"])
                prediction = float(first_row["Prediction"])

                date_value = pd.to_datetime(first_row["Date"])
                date = date_value.strftime("%Y-%m-%d")

                size, type_store = get_store_info(store)
                if size is None or type_store is None:
                    raise ValueError(f"Store introuvable dans stores.csv : {store}")

                hist_sd = history[
                    (history["Store"] == store) &
                    (history["Dept"] == dept)
                ].copy().sort_values("Date")

                last_4 = hist_sd.tail(4)

                if len(last_4) < 4:
                    chart_labels = ["Prévision"]
                    chart_values = [float(round(prediction, 2))]
                else:
                    mean_4w = float(last_4["Weekly_Sales"].mean())
                    last_week = float(last_4["Weekly_Sales"].iloc[-1])

                    chart_labels = [
                        "Moy 4 semaines",
                        "Dernière semaine",
                        "Prévision"
                    ]
                    chart_values = [
                        float(mean_4w),
                        float(last_week),
                        float(round(prediction, 2))
                    ]

                form_data = {
                    "Store": store,
                    "Dept": dept,
                    "Date": date,
                    "Temperature": first_row.get("Temperature", ""),
                    "Fuel_Price": first_row.get("Fuel_Price", ""),
                    "CPI": first_row.get("CPI", ""),
                    "Unemployment": first_row.get("Unemployment", ""),
                    "Size": size,
                    "Type": type_store,
                    "MarkDown1": first_row.get("MarkDown1", ""),
                    "MarkDown2": first_row.get("MarkDown2", ""),
                    "MarkDown3": first_row.get("MarkDown3", ""),
                    "MarkDown4": first_row.get("MarkDown4", ""),
                    "MarkDown5": first_row.get("MarkDown5", ""),
                    "IsHoliday": str(int(first_row.get("IsHoliday", 0))),
                    "Black_Friday": str(int(first_row.get("Black_Friday", 0))),
                    "Thanksgiving": str(int(first_row.get("Thanksgiving", 0))),
                    "Xmas_Week": str(int(first_row.get("Xmas_Week", 0))),
                }

                return render_template(
                    "index.html",
                    prediction=round(prediction, 2),
                    form_data=form_data,
                    model_used="XGBoost",
                    error=None,
                    chart_labels=chart_labels,
                    chart_values=chart_values,
                    date=date
                )

            # ===============================
            # 2) MODE FORMULAIRE MANUEL
            # ===============================
            if not request.form.get("Store"):
                return render_template(
                    "index.html",
                    error="Veuillez remplir le formulaire ou uploader un fichier",
                    form_data={},
                    chart_labels=[],
                    chart_values=[]
                )

            store = int(request.form["Store"])
            dept = int(request.form["Dept"])
            date = request.form["Date"]

            temperature = float(request.form["Temperature"])
            fuel_price = float(request.form["Fuel_Price"])
            cpi = float(request.form["CPI"])
            unemployment = float(request.form["Unemployment"])

            # Récupération automatique des infos fixes du store
            size, type_store = get_store_info(store)

            if size is None or type_store is None:
                return render_template(
                    "index.html",
                    error="Store invalide : informations magasin introuvables",
                    form_data=request.form,
                    chart_labels=[],
                    chart_values=[]
                )

            markdown1 = float(request.form["MarkDown1"])
            markdown2 = float(request.form["MarkDown2"])
            markdown3 = float(request.form["MarkDown3"])
            markdown4 = float(request.form["MarkDown4"])
            markdown5 = float(request.form["MarkDown5"])

            is_holiday = int(request.form["IsHoliday"])

            black_friday = int(request.form["Black_Friday"])
            thanksgiving = int(request.form["Thanksgiving"])
            xmas_week = int(request.form["Xmas_Week"])

            date_input = pd.to_datetime(date)

            # ===============================
            # Validation métier
            # ===============================
            if store not in history["Store"].unique():
                return render_template(
                    "index.html",
                    error="Store invalide",
                    form_data=request.form,
                    chart_labels=[],
                    chart_values=[]
                )

            valid_depts = history[history["Store"] == store]["Dept"].unique()

            if dept not in valid_depts:
                return render_template(
                    "index.html",
                    error="Deptartement invalide",
                    form_data=request.form,
                    chart_labels=[],
                    chart_values=[]
                )

            hist_sd = history[
                (history["Store"] == store) & (history["Dept"] == dept)
            ].copy()

            if hist_sd.empty:
                return render_template(
                    "index.html",
                    error="Pas d'historique",
                    form_data=request.form,
                    chart_labels=[],
                    chart_values=[]
                )

            max_date = hist_sd["Date"].max()
            next_allowed_date = max_date + pd.Timedelta(days=7)

            if date_input != next_allowed_date:
                return render_template(
                    "index.html",
                    error=f"Date non valide. La Semaine valide à prédire est : {next_allowed_date.strftime('%d/%m/%Y')}",
                    form_data=request.form,
                    chart_labels=[],
                    chart_values=[]
                )

            # ===============================
            # Prédiction
            # ===============================
            prediction = float(predict_sales(
                store=store,
                dept=dept,
                date_str=date,
                temperature=temperature,
                fuel_price=fuel_price,
                cpi=cpi,
                unemployment=unemployment,
                is_holiday=is_holiday,
                markdown1=markdown1,
                markdown2=markdown2,
                markdown3=markdown3,
                markdown4=markdown4,
                markdown5=markdown5,
                size=size,
                type_store=type_store,
                black_friday=black_friday,
                thanksgiving=thanksgiving,
                xmas_week=xmas_week
            ))

            # ===============================
            # Graphique
            # ===============================
            hist_sd = hist_sd.sort_values("Date")
            last_4 = hist_sd.tail(4)

            if len(last_4) < 4:
                chart_labels = ["Prévision"]
                chart_values = [float(round(prediction, 2))]
            else:
                mean_4w = float(last_4["Weekly_Sales"].mean())
                last_week = float(last_4["Weekly_Sales"].iloc[-1])

                chart_labels = ["Moy 4 semaines", "Dernière semaine", "Prévision"]
                chart_values = [
                    float(mean_4w),
                    float(last_week),
                    float(round(prediction, 2))
                ]

            form_data = request.form.to_dict()
            form_data["Size"] = size
            form_data["Type"] = type_store

            return render_template(
                "index.html",
                prediction=round(prediction, 2),
                form_data=form_data,
                model_used="XGBoost",
                error=None,
                chart_labels=chart_labels,
                chart_values=chart_values,
                date=date
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=f"Une erreur est survenue : {str(e)}",
                form_data=request.form if request.method == "POST" else {},
                chart_labels=[],
                chart_values=[]
            )

    return render_template(
        "index.html",
        form_data={},
        error=None,
        chart_labels=[],
        chart_values=[]
    )


if __name__ == "__main__":
    app.run(debug=True)