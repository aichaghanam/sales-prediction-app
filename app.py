from flask import Flask, render_template, request
from predict_utils import predict_sales, history
import pandas as pd
import json
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        try:
            # ===============================
            # Récupération des inputs utiles
            # ===============================

            store = int(request.form["Store"])
            dept = int(request.form["Dept"])
            date = request.form["Date"]

            is_holiday = int(request.form["IsHoliday"])
            markdown4 = float(request.form["MarkDown4"])
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
                    error="Store invalide : ce magasin n'existe pas",
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
            # Prédiction LightGBM
            # ===============================

            prediction = predict_sales(
                store=store,
                dept=dept,
                date_str=date,
                is_holiday=is_holiday,
                markdown4=markdown4,
                size=size,
                type_store=type_store,
                black_friday=black_friday,
                thanksgiving=thanksgiving,
                xmas_week=xmas_week
            )
            # ===== Graphique amélioré : historique récent + prévision =====
            hist_sd = hist_sd.sort_values("Date").copy()

            last_4 = hist_sd.tail(4)

            if len(last_4) < 4:
                chart_labels = ["Prévision"]
                chart_values = [round(prediction, 2)]
            else:
                mean_4w = round(last_4["Weekly_Sales"].mean(), 2)
                last_week = round(last_4["Weekly_Sales"].iloc[-1], 2)

                chart_labels = [
                    "Moy 4 semaines",
                    "Dernière semaine",
                    "Prévision"
                ]
                chart_values = [
                    mean_4w,
                    last_week,
                    round(prediction, 2)
                ]

        
            return render_template(
                "index.html",
                prediction=round(prediction, 2),
                form_data=request.form,
                model_used="LightGBM",
                error=None,
                chart_labels=chart_labels,
                chart_values=chart_values,
                date=date
            )

        except ValueError:
            return render_template(
                "index.html",
                error="Entrée invalide : merci de saisir uniquement des nombres dans les champs numériques.",
                form_data=request.form,
                chart_labels="[]",
                chart_values="[]"
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=f"Une erreur est survenue : {str(e)}",
                form_data=request.form,
                chart_labels="[]",
                chart_values="[]"
            )

  
    return render_template(
        "index.html",
        form_data={},
        error=None,
        chart_labels="[]",
        chart_values="[]"
    )


if __name__ == "__main__":
    app.run(debug=True)