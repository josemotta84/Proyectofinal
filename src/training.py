import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. CARGA DE DATOS (URL RAW)
data_url = "https://raw.githubusercontent.com/josemotta84/Proyectofinal/main/data_airbnb_sample.csv"

try:
    data_airbnb = pd.read_csv(data_url)
    print(f"✅ Datos cargados desde GitHub: {len(data_airbnb)} filas.")
except:
    data_airbnb = pd.read_csv("data_airbnb_sample.csv") # Fallback local
    print("⚠️ Cargando datos desde archivo local.")

# 2. PREPARACIÓN (Solo una vez para evitar KeyError)
features_num = ['accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'review_scores_location']
target = 'price'

# Filtramos columnas necesarias y eliminamos nulos
df_model = data_airbnb[features_num + ['room_type', target]].dropna().copy()

# 3. TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS
# Aquí 'room_type' se convierte en columnas numéricas y desaparece del índice
df_model = pd.get_dummies(df_model, columns=['room_type'], drop_first=True)

# 4. DEFINIR X e Y
X = df_model.drop(columns=[target])
y = df_model[target]

# 5. DIVISIÓN Y ESCALADO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. ENTRENAMIENTO DEL MODELO
print("Entrenando Random Forest... por favor espera.")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# 7. EVALUACIÓN
y_pred = rf_model.predict(X_test_scaled)
print(f"\n--- Resultados del Modelo ---")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# 8. GUARDAR MODELO Y ESCALADOR
joblib.dump(rf_model, 'rf_model_airbnb.joblib')
joblib.dump(scaler, 'scaler_airbnb.joblib')
print("✅ Archivos guardados: rf_model_airbnb.joblib y scaler_airbnb.joblib")

