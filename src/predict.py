import joblib
import pandas as pd

# Cargar el modelo guardado
modelo = joblib.load("../modelo_random_forest.pkl")

# Función para hacer predicciones en nuevos clientes
def predecir_churn(datos_cliente):
    df = pd.DataFrame([datos_cliente])  # Convertir en DataFrame
    prediccion = modelo.predict(df)  # Hacer predicción
    return "Churn" if prediccion[0] == 1 else "No Churn"

# Ejemplo de cliente nuevo
nuevo_cliente = {
    "Tenure Months": 24,
    "Monthly Charges": 75.0,
    "Total Charges": 1800.0,
    "InternetService_Fiber optic": 1,
    "Contract_Two year": 1,
    "PaymentMethod_Credit card (automatic)": 1
}

resultado = predecir_churn(nuevo_cliente)
print(f"✅ Predicción para el cliente: {resultado}")