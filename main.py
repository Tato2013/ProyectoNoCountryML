from fastapi import FastAPI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, column, Integer, Float, String, Text, Boolean, Date, DateTime, Time, MetaData, Table, text, insert, select
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import time
from decouple import Config, Csv

#**********************************
#Configurar variables de entorno
#**********************************
app=FastAPI()



class Config:
    def __init__(self):
        # Crea una instancia de Config de decouple
        self.config = Config()
        # Carga las variables de entorno desde el archivo .env
        self.config.read_dotenv()

    def get(self, key):
        # Obtiene el valor de la variable de entorno
        return self.config(key)

# Crear una instancia de la clase Config
config = Config()

# Obtener los valores de las variables de entorno
repository_value = config.get('REPOSITORY_VALUE')

#*********************************************************
#Ingreso a la base de datos                              *
#*********************************************************

# Obtén los valores de las variables de entorno
username = config('DB_USERNAME')
password = config('DB_PASSWORD')
host = config('DB_HOST')
port = config('DB_PORT')
database = config('DB_DATABASE')

#*******************************
# Construye la URL de conexión
#********************************
db_url = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
engine = create_engine(db_url)
historico = pd.read_sql('historico', engine)
#cerramos conexion una vez obtenidos los datos
engine.dispose()

historico['Fecha'] = historico['Date'].astype('int64') // 10**9
ultima_fecha = historico['Date'].max()  # Asegúrate de que 'Fecha' esté en formato de fecha
# Crear características para la predicción futura
nueva_fecha = ultima_fecha + pd.DateOffset(days=7)

#**********************************
#Crear funciones
#**********************************

def generar_caracteristicas_futuras(ultima_fecha):
    nueva_fecha = ultima_fecha + pd.DateOffset(days=7)
    # Ajusta esta lógica según tus necesidades
    nuevas_caracteristicas = pd.DataFrame({'Fecha': [nueva_fecha.timestamp() // 1e9]})
    return nuevas_caracteristicas


#**************************
#Se define consulta
#**************************

    
@app.get("/accion/{accion}")
def obtener_prediccion_RandonForest(accion: str) -> dict:
    df_accion = historico[historico['Ticket'] == accion]
    
    # Obtener características (X) y variable objetivo (y) para la acción actual
    X = df_accion[['Fecha']]  # Agrega las características relevantes
    y = df_accion['Close']
    
    # División de datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicialización del modelo de Random Forest
    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    
    # Entrenamiento del modelo
    modelo_rf.fit(X_train, y_train)
    
    # Almacenar el modelo en el diccionario
    ultima_fecha = historico['Date'].max()+ timedelta(days=7)

    # Predicciones en el conjunto de prueba
    y_pred = modelo_rf.predict(X_test)
    
   
    # Evaluación del rendimiento del modelo
    
    r2 = r2_score(y_test, y_pred)
    return {
        "accion": accion,
        "prediccion_futura":y_pred[0],
        'Fecha': ultima_fecha,
        'R2':r2
    }