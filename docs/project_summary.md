# 📌 Resumen del Proyecto

Este repositorio contiene el análisis de **churn en telecomunicaciones** mediante **Machine Learning**. Se entrenaron tres modelos:
- **Random Forest (mejor modelo seleccionado)**
- **K-Nearest Neighbors (KNN)**
- **Regresión Logística**

Se usó **SMOTE para balanceo de clases**, **GridSearchCV para optimización de hiperparámetros**, y se evaluaron **métricas como Accuracy, Precision, Recall y AUC-ROC**.

### 📊 Resultados Finales

| Modelo           | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|----------------|----------|----------|--------|---------|---------|
| **Random Forest** | 0.8012 | 0.7742   | 0.8410 | 0.8046  | 0.7985  |
| **KNN**         | 0.7402 | 0.7120   | 0.8725 | 0.8085  | 0.7501  |
| **Regresión Logística** | 0.7405 | 0.7201   | 0.7520 | 0.7420  | 0.7450  |

🔹 **El modelo final recomendado es Random Forest.**