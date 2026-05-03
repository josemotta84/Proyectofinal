import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Configuración de rutas (Usa nombres de archivos locales para modelos)
MODEL_PATH = 'rf_model_airbnb.joblib'
SCALER_PATH = 'scaler_airbnb.joblib'
# URL Raw para los datos
DATA_PATH = "https://raw.githubusercontent.com/josemotta84/Proyectofinal/main/data_airbnb_sample.csv"

def validate_model():
    print("Iniciando validación...")
    
    # Cargar modelo y escalador (deben existir localmente tras ejecutar training.py)
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Modelo y escalador cargados correctamente.")
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos .joblib. Ejecuta primero src/training.py")
        return

    # Cargar datos desde GitHub
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Datos de validación cargados: {len(df)} filas.")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return
    
    # 2. Preprocesamiento (Idéntico al entrenamiento)
    features_num = ['accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'review_scores_location']
    target = 'price'
    
    # Selección de columnas y limpieza
    df_val = df[features_num + ['room_type', target]].dropna().copy()
    
    # Encoding
    df_val = pd.get_dummies(df_val, columns=['room_type'], drop_first=True)
    
    # Separar X e y
    X_val = df_val.drop(columns=[target])
    y_val = df_val[target]
    
    # 3. Escalado (Fundamental para que coincida con el entrenamiento)
    X_val_scaled = scaler.transform(X_val)
    
    # 4. Predicción
    y_pred = model.predict(X_val_scaled)
    
    # 5. Cálculo de Métricas
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"\n--- REPORTE DE VALIDACIÓN ---")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # 6. Visualización
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.3, color='green')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r')
    plt.xlabel('Precio Real')
    plt.ylabel('Precio Predicho')
    plt.title('Validación Airbnb: Real vs. Predicho')
    plt.show()

if __name__ == "__main__":
    validate_model()