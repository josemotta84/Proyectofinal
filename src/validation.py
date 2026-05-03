import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Configuración de rutas y carga
MODEL_PATH = "https://github.com/josemotta84/Proyectofinal/rf_model_airbnb.joblib"
DATA_PATH = "https://github.com/josemotta84/Proyectofinal/data_airbnb_test_sample.csv"

def validate_model():
    print("Cargando modelo y datos para validación...")
    
    # Cargar modelo
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: No se encontró el archivo del modelo. ¿Ya lo guardaste?")
        return

    # Cargar datos
    df = pd.read_csv(DATA_PATH)
    
    # 2. Preprocesamiento (Debe ser idéntico al de entrenamiento)
    features_num = ['accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'review_scores_location']
    target = 'price'
    
    df_val = df[features_num + ['room_type', target]].copy().dropna()
    
    # Encoding (One-Hot)
    df_val = pd.get_dummies(df_val, columns=['room_type'], drop_first=True)
    
    # Definir X e y
    X_val = df_val.drop(columns=[target])
    y_val = df_val[target]
    
    # 3. Predicción
    y_pred = model.predict(X_val)
    
    # 4. Cálculo de Métricas
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    
    print(f"\n--- REPORTE DE VALIDACIÓN ---")
    print(f"MAE (Error Absoluto Medio): ${mae:.2f}")
    print(f"RMSE (Error Cuadrático Medio): ${rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # 5. Visualización: Real vs. Predicho
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.3, color='blue')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', linewidth=2)
    plt.xlabel('Precio Real')
    plt.ylabel('Precio Predicho')
    plt.title('Validación: Real vs. Predicho')
    plt.show()

if __name__ == "__main__":
    validate_model()