from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inicializar la API
app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load("models/random_forest.pkl")

# Definir las columnas que el modelo espera
columnas_modelo = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'InternetService_Fiber optic', 
                   'Contract_Two year', 'PaymentMethod_Credit card (automatic)']

# Función de preprocesamiento (ajustar datos para que coincidan con el modelo)
def preprocesar_datos(datos):
    df = pd.DataFrame([datos])  # Convertir en DataFrame

    # 🔹 Definir las columnas esperadas por el modelo
    columnas_esperadas = [
        "Count", "Zip Code", "Latitude", "Longitude", "Tenure Months",
        "Monthly Charges", "Total Charges", "CLTV"
    ]

    # 🔹 Asegurar que todas las columnas esperadas estén presentes
    for col in columnas_esperadas:
        if col not in df:
            df[col] = 0  # Agregar columna faltante con valor 0

    # 🔹 Devolver el DataFrame con el orden correcto
    return df[columnas_esperadas]

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        datos = request.get_json()
        print("📩 Datos recibidos en la API:", datos)  # 🟢 Verificar si recibe datos

        if not datos:
            print("⚠️ No se recibieron datos.")
            return jsonify({"error": "No se recibieron datos"}), 400

        df_procesado = preprocesar_datos(datos)
        prediccion = modelo.predict(df_procesado)[0]

        print("✅ Predicción realizada:", prediccion)  # 🟢 Verificar si predice bien

        return jsonify({"Churn_Prediction": int(prediccion)})

    except Exception as e:
        print("❌ Error en la API:", str(e))  # 🔴 Imprimir errores
        return jsonify({"error": str(e)}), 500
    
# Iniciar la API en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)  # Cambia a un puerto disponible