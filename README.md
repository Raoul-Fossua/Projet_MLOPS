# 🏦 Projet MLOps — Prédiction du risque de défaut de prêt personnel

## 🎯 Objectif du projet
L’objectif de ce projet est de prédire la **probabilité de défaut de paiement** d’un client sur un prêt personnel, afin d’aider la banque à mieux gérer son risque de crédit et à optimiser ses décisions d’octroi.

---

## 👥 Membres du groupe 

Naya Maoudana KARE
Raoul FOSSUA
Issa BICHARA
Anais DELIGNY

## 🧠 Démarche MLOps

Le projet suit une approche **MLOps end-to-end** incluant :

1. **Préparation des données**
   - Nettoyage, imputation des valeurs manquantes, encodage, normalisation.
2. **Feature Engineering**
   - Création de nouvelles variables (ex : ratio `loan_to_income`).
3. **Modélisation**
   - Entraînement et comparaison de 3 modèles :
     - Régression Logistique
     - Arbre de Décision
     - Forêt Aléatoire
4. **Suivi des expériences avec MLflow**
   - Un modèle = un *experiment*  
   - Une itération = un *run*  
   - Tracking des hyperparamètres, métriques et artefacts.
5. **Déploiement**
   - Application web **Streamlit** pour tester le modèle.
   - Déploiement sur le **Cloud (Google Cloud Run)** via **CI/CD GitHub Actions**.

---

## 🏗️ Structure du projet

```
mlops-credit-default/
├─ config/
│   ├─ config.yaml
│   └─ features.yaml
├─ data/
│   ├─ raw/
│   ├─ interim/
│   └─ processed/
├─ src/
│   ├─ data/
│   ├─ features/
│   ├─ models/
│   ├─ utils/
│   └─ app/
├─ mlruns/
├─ notebooks/
├─ Dockerfile
├─ Makefile
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Installation et exécution locale

### 1️⃣ Installation des dépendances
```bash
python -m venv .venv
source .venv/bin/activate    # (sous Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2️⃣ Préparation des données
```bash
python -m src.data.make_dataset --config config/config.yaml
python -m src.features.build_features --config config/config.yaml --features config/features.yaml
```

### 3️⃣ Lancer MLflow pour suivre les expériences
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

### 4️⃣ Entraîner les modèles
```bash
python -m src.models.train --config config/config.yaml --features config/features.yaml --experiment_suffix dev
```

---

## 💻 Application Streamlit

### Lancer l’application localement :
```bash
MLFLOW_TRACKING_URI=file:./mlruns MODEL_NAME=credit_pd_rf MODEL_STAGE=Production \
streamlit run src/app/streamlit_app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501)

L’application permet :
- de saisir manuellement les informations d’un client pour obtenir sa probabilité de défaut ;
- de charger un fichier CSV pour effectuer des prédictions en lot.

---

## ☁️ Déploiement Cloud

Le déploiement est automatisé via **GitHub Actions** :
- Build de l’image Docker
- Push sur **Google Container Registry**
- Déploiement sur **Google Cloud Run**

### Secrets nécessaires sur GitHub :
- `GCP_PROJECT`  
- `GCP_REGION`  
- `GCP_SA_KEY`  
- `MLFLOW_TRACKING_URI`

---

## 📊 Suivi avec MLflow

- Un *experiment* par modèle (`credit_default_logreg`, `credit_default_rf`, etc.)
- Un *run* par essai/hyperparamètre
- Métriques loggées :
  - `roc_auc`, `pr_auc`, `brier`, `logloss`
- Les modèles sont sauvegardés dans le *Model Registry* pour être promus en **Staging** ou **Production**.

---

## 🧩 Technologies utilisées

| Composant | Outil |
|------------|--------|
| Langage | Python 3.11 |
| Data | pandas, numpy, scikit-learn |
| Tracking | MLflow |
| Interface | Streamlit |
| CI/CD | GitHub Actions |
| Déploiement | Google Cloud Run |
| Conteneurisation | Docker |

---

---

## 🏁 Conclusion

Ce projet illustre une **pipeline MLOps complète**, combinant :
- préparation des données,
- modélisation supervisée,
- suivi d’expériences MLflow,
- et déploiement automatisé sur le cloud.

Un projet industrialisé, traçable et reproductible. 🚀
