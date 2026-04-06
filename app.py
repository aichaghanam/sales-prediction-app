from flask import Flask, render_template, request
from predict_utils import predict_sales, history
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        try:
            # ===============================
            # Récupération des inputs
            # ===============================

            store = int(request.form["Store"])
            dept = int(request.form["Dept"])
            date = request.form["Date"]

            is_holiday = int(request.form["IsHoliday"])

            temperature = float(request.form["Temperature"])
            fuel_price = float(request.form["Fuel_Price"])

            markdown1 = float(request.form["MarkDown1"])
            markdown2 = float(request.form["MarkDown2"])
            markdown3 = float(request.form["MarkDown3"])
            markdown4 = float(request.form["MarkDown4"])
            markdown5 = float(request.form["MarkDown5"])

            cpi = float(request.form["CPI"])
            unemployment = float(request.form["Unemployment"])
            size = int(request.form["Size"])

            type_store = request.form["Type"]

            black_friday = int(request.form["Black_Friday"])
            thanksgiving = int(request.form["Thanksgiving"])
            xmas_week = int(request.form["Xmas_Week"])

            date_input = pd.to_datetime(date)

            # ===============================
            # Vérification Store
            # ===============================

            if store not in history["Store"].unique():
                return render_template(
                    "index.html",
                    error="Store invalide : ce magasin n'existe pas dans les données.",
                    form_data=request.form
                )

            # ===============================
            # Vérification Dept pour ce Store
            # ===============================

            valid_depts = history[history["Store"] == store]["Dept"].unique()

            if dept not in valid_depts:
                return render_template(
                    "index.html",
                    error=f"Dept invalide : le département {dept} n'existe pas pour le Store {store}.",
                    form_data=request.form
                )

            # ===============================
            # Vérification date par Store/Dept
            # ===============================

            hist_sd = history[
                (history["Store"] == store) &
                (history["Dept"] == dept)
            ].copy()

            if hist_sd.empty:
                return render_template(
                    "index.html",
                    error="Aucun historique disponible pour ce couple Store/Dept.",
                    form_data=request.form
                )

            max_date = hist_sd["Date"].max()
            next_allowed_date = max_date + pd.Timedelta(days=7)

            if date_input != next_allowed_date:
                return render_template(
                    "index.html",
                    error=(
                        f"La seule semaine autorisée à prédire pour ce couple "
                        f"(Store/Dept) est : {next_allowed_date.strftime('%d/%m/%Y')}"
                    ),
                    form_data=request.form
                )

            # ===============================
            # Choix du modèle
            # ===============================

            model_type = request.form.get("model_type", "rf")

            # ===============================
            # Prédictions des deux modèles
            # ===============================

            prediction_rf = predict_sales(
                store, dept, date, is_holiday,
                temperature, fuel_price,
                markdown1, markdown2, markdown3, markdown4, markdown5,
                cpi, unemployment, size,
                type_store,
                black_friday, thanksgiving, xmas_week,
                model_type="rf"
            )

            prediction_xgb = predict_sales(
                store, dept, date, is_holiday,
                temperature, fuel_price,
                markdown1, markdown2, markdown3, markdown4, markdown5,
                cpi, unemployment, size,
                type_store,
                black_friday, thanksgiving, xmas_week,
                model_type="xgb"
            )

            # ===============================
            # Sélection du modèle affiché
            # ===============================

            if model_type == "rf":
                prediction = prediction_rf
                model_used = "Random Forest Regressor"
            else:
                prediction = prediction_xgb
                model_used = "Extreme Gradient Boosting (XGBoost)"

            return render_template(
                "index.html",
                prediction=round(prediction, 2),
                form_data=request.form,
                model_used=model_used,
                prediction_rf=round(prediction_rf, 2),
                prediction_xgb=round(prediction_xgb, 2),
                error=None
            )

        except ValueError:
            return render_template(
                "index.html",
                error="Entrée invalide : merci de saisir uniquement des nombres dans les champs numériques.",
                form_data=request.form
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=f"Une erreur est survenue : {str(e)}",
                form_data=request.form
            )

    # ===============================
    # GET
    # ===============================

    return render_template("index.html", form_data={}, error=None)


if __name__ == "__main__":
    app.run(debug=True)