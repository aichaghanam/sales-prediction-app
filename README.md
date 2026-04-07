# Walmart Weekly Sales Forecast

## Description

Ce projet est une application de prévision des ventes hebdomadaires pour Walmart.  
Il permet de prédire les ventes d’un département spécifique dans un magasin donné pour une semaine future à l’aide d’un modèle LightGBM.

Le projet couvre tout le pipeline :
- préparation des données  
- feature engineering  
- entraînement du modèle  
- déploiement via une application web Flask  

---

## Objectif

Prédire les ventes hebdomadaires (Weekly Sales) pour un couple Store / Department.

Cas d’usage :
- gestion des stocks  
- planification des promotions  
- aide à la décision en retail  

---

## Application en ligne

L’application est accessible ici :  
https://sales-prediction-app-942r.onrender.com

---

## Structure du projet

```text
.
├── app.py
├── predict_utils.py
├── prepare_data.py
├── train_lgbm_model.py
├── requirements.txt
├── templates/
│   └── index.html
├── data/
│   ├── train.csv
│   ├── features.csv
│   └── stores.csv
├── model_lgbm.pkl
├── features_final.pkl
├── encoders.pkl
├── history.pkl
```

---

## Fonctionnement

Pipeline global :

1. Chargement des données  
2. Feature engineering  
3. Entraînement du modèle  
4. Sauvegarde des artefacts  
5. Prédiction via application web  

---

## Feature Engineering

Features utilisées :

- Variables temporelles : Year, Month, Week  
- Historique :
  - Lag_1, Lag_2, Lag_4  
  - Rolling mean / std  
- Encodages :
  - Store_enc  
  - Dept_enc  
  - Store_Dept_enc  
- Variables métier :
  - Size  
  - Type (one-hot)  
- Événements :
  - Black Friday  
  - Thanksgiving  
  - Xmas Week  

---

## Modèle

- Algorithme : LightGBM  
- Transformation cible : log(Weekly Sales)  
- Validation : TimeSeriesSplit  
- Métriques : WMAE, MAE, RMSE, R²  

---

## Prédiction

Étapes :

1. Validation des entrées utilisateur  
2. Vérification Store / Dept / Date  
3. Calcul des features dynamiques :
   - lags  
   - rolling statistics  
   - encodings  
   - date features  
4. Construction du dataset  
5. Prédiction avec LightGBM  
6. Transformation inverse (exp)  

---

## Contraintes métier

- Store doit exister  
- Dept doit appartenir au Store  
- Date = semaine suivante disponible  

---

## Installation locale

```bash
git clone https://github.com/your-username/walmart-forecast.git
cd walmart-forecast
pip install -r requirements.txt
```

---

## Lancement local

```bash
python app.py
```

Accès :  
http://127.0.0.1:5000/

---

## Déploiement

L’application est déployée sur Render en tant que Web Service.

Configuration :

- Environnement : Python  
- Commande de démarrage :

```bash
gunicorn app:app
```

- Fichier de dépendances :  
requirements.txt  

- Artefacts chargés au runtime :
  - model_lgbm.pkl  
  - features_final.pkl  
  - encoders.pkl  
  - history.pkl  

---

## Dépendances principales

- Flask  
- pandas  
- numpy  
- scikit-learn  
- lightgbm  
- gunicorn  

---

## Améliorations possibles

- API REST pour prédictions batch  
- Dashboard interactif  
- Dockerisation  
- Automatisation des features événementielles  
- Déploiement cloud avancé  

---

## Licence

MIT License
