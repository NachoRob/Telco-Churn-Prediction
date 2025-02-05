# 📌 Entrenamiento de Modelos para la Predicción de Churn

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 📌 1. Cargar datos preprocesados
X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

# 📌 2. Configurar modelos y grid de hiperparámetros
param_grid_rf = {"n_estimators": [50, 100], "max_depth": [10, 20], "min_samples_split": [2, 5]}
param_grid_knn = {"n_neighbors": [3, 5], "metric": ["manhattan", "euclidean"], "weights": ["uniform", "distance"]}
param_grid_logreg = {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["saga"]}

# 📌 3. Entrenamiento con GridSearchCV

# Random Forest
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring="f1", n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
joblib.dump(best_rf, "../models/random_forest.pkl")

# KNN
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring="f1", n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
joblib.dump(best_knn, "../models/knn.pkl")

# Regresión Logística
grid_logreg = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42), param_grid_logreg, cv=5, scoring="f1", n_jobs=-1)
grid_logreg.fit(X_train, y_train)
best_logreg = grid_logreg.best_estimator_
joblib.dump(best_logreg, "../models/logistic_regression.pkl")

print("✅ Modelos entrenados y guardados.")ß