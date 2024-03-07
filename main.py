from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, column, Integer, Float, String, Text, Boolean, Date, DateTime, Time, MetaData, Table, text, insert, select
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import time
import os
from dotenv import find_dotenv , load_dotenv
from typing import Dict

#**********************************
#Configurar variables de entorno
#**********************************
app=FastAPI()


# Carga las variables de entorno desde el archivo .env
dotenv_patch=find_dotenv()

load_dotenv(dotenv_patch)
#*********************************************************
# Ingreso a la base de datos                              *
#*********************************************************

# Obtén los valores de las variables de entorno
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_DATABASE')

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
acciones = historico['Ticket'].unique()

#**********************************
#Crear funciones
#**********************************

def rendimiento_mensual(accion,fecha1,ultima_fecha):
    df_accion = historico[(historico['Ticket'] == accion) & (historico['Date'] >= fecha1) & (historico['Date'] <= ultima_fecha)]

    precio_cierre_primero = df_accion[df_accion['Date'] == fecha1]['Close'].values[0]
    precio_cierre_ultimo = df_accion[df_accion['Date'] == ultima_fecha]['Close'].values[0]

    # Calcular el rendimiento mensual
    rendimiento = ((precio_cierre_ultimo - precio_cierre_primero) / precio_cierre_primero) * 100

    return rendimiento

def encontrar_primer_dia_habil(accion, ultima_fecha):
    fecha_actual = ultima_fecha - timedelta(days=30)
    
    while True:
        # Verificar si el día actual es sábado o domingo
        while fecha_actual.weekday() in [5, 6]:
            fecha_actual += timedelta(days=1)

        # Verificar si la fecha actual está en el DataFrame
        if not historico[(historico['Ticket'] == accion) & (historico['Date'] == fecha_actual)].empty:
            return fecha_actual

        # Si la fecha actual no está en el DataFrame, sumar un día
        fecha_actual += timedelta(days=1)

#**************************
#Se define consulta
#**************************

#___________________________________________________________________
#consulta RandomForest
#___________________________________________________________________
@app.get("/accion/rf/{accion}")
def obtener_prediccion_RandonForest(accion: str) -> dict:
    # Convertir la acción a mayúsculas
    accion = accion.upper()
    #Verificar datos
    if accion not in acciones:
        raise HTTPException(status_code=404, detail=f'La acción {accion} no está en la lista de acciones disponibles. Elija entre: {acciones}')
    df_accion = historico[historico['Ticket'] == accion]
    
    # Obtener características (X) y variable objetivo (y) para la acción actual
    X = df_accion[['Fecha']]  
    y = df_accion['Close']
    
    # División de datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicialización del modelo de Random Forest
    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    
    # Entrenamiento del modelo
    modelo_rf.fit(X_train, y_train)
    
    # Almacenar el modelo en el diccionario
    ultima_fecha = historico['Date'].max()+ timedelta(days=7)
    ultima_fecha = historico['Date'].max()
    fecha_prediccion = ultima_fecha + timedelta(days=7)
    fecha_prediccion=fecha_prediccion
    # Crear un nuevo DataFrame con la fecha de predicción
    nueva_entrada = pd.DataFrame({'Fecha': [fecha_prediccion]})
    X_prediccion = nueva_entrada[['Fecha']].astype('int64') // 10**9

# Realizar la predicción para la fecha futura
    precio_predicho = modelo_rf.predict(X_prediccion)
    
    y_pred_test = modelo_rf.predict(X_test)
    fecha_formateada = fecha_prediccion.strftime('%Y-%m-%d')
    # Evaluar el rendimiento del modelo utilizando r2_score en el conjunto de prueba
    r2 = r2_score(y_test, y_pred_test)
    return {
        "accion": accion,
        "prediccion_futura":precio_predicho[0],
        'Fecha': fecha_formateada,
        'R2':r2
    }
    
#_____________________________________________________________
#Consulta Regresion Lineal
#______________________________________________________________  
  
@app.get("/accion/regresion/{accion}")
def obtener_prediccion_RegresionLineal(accion: str) -> dict:
    # Convertir la acción a mayúsculas
    accion = accion.upper()
    #Verificar datos
    if accion not in acciones:
        raise HTTPException(status_code=404, detail=f'La acción {accion} no está en la lista de acciones disponibles. Elija entre: {acciones}')
    df_accion = historico[historico['Ticket'] == accion]
    X = df_accion[['Fecha']]  
    y = df_accion['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    
    # Inicialización del modelo de regresión lineal
    modelo_regresion = LinearRegression()
    
    # Entrenamiento del modelo
    modelo_regresion.fit(X_train, y_train)
    
    # Almacenar el modelo en el diccionario
    ultima_fecha = historico['Date'].max()+ timedelta(days=7)
    ultima_fecha = historico['Date'].max()
    fecha_prediccion = ultima_fecha + timedelta(days=7)
    fecha_prediccion=fecha_prediccion
    # Crear un nuevo DataFrame con la fecha de predicción
    nueva_entrada = pd.DataFrame({'Fecha': [fecha_prediccion]})
    X_prediccion = nueva_entrada[['Fecha']].astype('int64') // 10**9

# Realizar la predicción para la fecha futura
    precio_predicho = modelo_regresion.predict(X_prediccion)
    
    y_pred_test = modelo_regresion.predict(X_test)
   
    # Evaluación del rendimiento del modelo
    fecha_formateada = fecha_prediccion.strftime('%Y-%m-%d')
    r2 = r2_score(y_test, y_pred_test)
    return {
        "accion": accion,
        "prediccion_futura":precio_predicho[0],
        'Fecha': fecha_formateada,
        'R2':r2
    }
    
#_____________________________________________________________
#Consulta comparacion de acciones rendiemiento semanal
#______________________________________________________________   
  
@app.get("/accion/comparacion")
def comparar_rendimiento(accion1: str, accion2: str) -> dict:
    # Convertir las acciones a mayúsculas
    accion1, accion2 = accion1.upper(), accion2.upper()

    #Verificar datos
    if accion1 not in acciones or accion2 not in acciones:
        raise HTTPException(status_code=404, detail=f'Una o ambas acciones no están en la lista de acciones disponibles. Elija entre: {acciones}')
    ultima_fecha = historico['Date'].max()
    if isinstance(ultima_fecha, str):
        ultima_fecha = pd.to_datetime(ultima_fecha)
    # Obtener el primer y último día del mes actual
    fecha_inicio=encontrar_primer_dia_habil(accion1,ultima_fecha)
    #Calcular el rendimiento de cada accion
    rendimiento1=rendimiento_mensual(accion1,fecha_inicio,ultima_fecha)
    rendimiento2=rendimiento_mensual(accion2,fecha_inicio,ultima_fecha)
    
    resultado_accion1 = {
        "accion": accion1,
        "periodo": f"{fecha_inicio} - {ultima_fecha}",
        "rendimiento_mensual%": rendimiento1
    }

    resultado_accion2 = {
        "accion": accion2,
        "periodo": f"{fecha_inicio} - {ultima_fecha}",
        "rendimiento_mensual%": rendimiento2
    }

    # Retornar ambos diccionarios en un diccionario principal
    return {"accion1": resultado_accion1, "accion2": resultado_accion2}
#_____________________________________________________________
#Consulta Rendimiento Semanal de la accion
#______________________________________________________________ 

@app.get("/accion/RendimeintoSemanal/{accion}")
def Rendimiento_ultima_Semana(accion: str) -> dict:
    # Convertir la acción a mayúsculas
    accion = accion.upper()
    
    if accion not in acciones:
        raise HTTPException(status_code=404, detail=f'La acción {accion} no está en la lista de acciones disponibles. Elija entre: {acciones}')
    df_accion = historico[historico['Ticket'] == accion]
    df=df_accion.tail(7)   
    df['Rendimiento_diario'] = (df['Close'] - df['Open']) / df['Open'] * 100

    
    #Creo un diccionario para mostrar los resultados
    resultado = {}

    # Crear diccionario
    for index, row in df.iterrows():
        fecha = str(row['Date'])
        rendimiento_diario = row['Rendimiento_diario']
        valor_cierre = row['Close']

        resultado[fecha] = {
            'Rendimiento_diario%': rendimiento_diario,
            'Valor_cierre': valor_cierre
        }

    return resultado

#_____________________________________________________________
#Consulta Ultimos datos ingreasados
#______________________________________________________________ 

@app.get("/accion/informacion-ultimo-dia/{accion}")
def informacion_ultimo_dia(accion: str) -> dict:
    accion = accion.upper()
    #Verificar datos
    if accion not in acciones:
        raise HTTPException(status_code=404, detail=f'La acción {accion} no está en la lista de acciones disponibles. Elija entre: {acciones}')
    
    # Filtrar el DataFrame para obtener la información del último día para la acción ingresada
    df_accion = historico[historico['Ticket'] == accion]
    
    # Verificar si hay datos para la acción ingresada
    if df_accion.empty:
        raise HTTPException(status_code=404, detail=f'No hay información disponible para la acción {accion}.')
    
    # Obtener la información del último día
    info_ultimo_dia = df_accion.iloc[-1].to_dict()
    
    return info_ultimo_dia
#-------------------------------------------------------------------------------------------------------------------      