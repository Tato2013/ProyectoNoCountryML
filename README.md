# Análisis de Tendencias Financieras en Acciones



## Introducción 

Bienvenido al proyecto de simulación de consultoría financiera GROWTREND, donde se lleva a cabo un análisis para orientar futuras inversiones. En esta sección, presentaremos un modelo de Machine Learning 
diseñado para comprender posibles tendencias futuras mediante estimaciones basadas en el estudio de los últimos 5 años de datos de acciones.

## Recursos

En este proyecto, se ha utilizado un modelo de Random Forest para prever tendencias futuras. Aunque Random Forest no es la opción óptima para este caso específico, existen alternativas más adecuadas,
como modelos especializados en series temporales o la combinación de varios modelos.También se puede considerar la integración de información externa, como noticias, para analizar la reacción del mercado
y los sentimientos de los accionistas e inversores.

## API 
El modelo se entrena utilizando los precios históricos de las siguientes acciones:

['AAPL' 'GOOG' 'MSFT' 'NVDA' 'AMZN' 'NFLX' 'TSLA' 'META' 'AMD' 'XOM' 'UBER' 'QCOM' 'COIN' 'KO' 'BAC' 'HD' 'PYPL' 'JPM' 'UNH' 'V']

Se Puede ingresar al mismo por el siguente [Link](https://proyectonocountryml.onrender.com/docs)

La API cuenta con las siguentes consultas:

### Modelo RandomForest

- **Descripción:** Este modelo, entrenado con datos limitados, predice un posible precio para la próxima semana. Dada la variabilidad en las acciones, la precisión puede ser limitada.

### Modelo de Regresión Lineal

- **Descripción:** Similar al modelo anterior, este utiliza regresión lineal para predecir un posible precio a una semana. También puede servir como base para un modelo de ensamble.

### Comparación del Rendimiento Mensual de Dos Acciones

- **Descripción:** Permite comparar el rendimiento en porcentaje de dos acciones para el último mes transcurrido.

### Rendimiento Diario de la Última Semana

- **Descripción:** Calcula el rendimiento diario en porcentaje y devuelve los resultados para la última semana de la acción consultada.

### Informacion del ultimo dia

- **Descripción:** Muestra la información de la acción seleccionada para el último día registrado en la base de datos.


### Consideracion

La base de datos es gratuita y el proyecto solo tiene 3 meses de funcionalidad que es lo que nos permite render utilizar de forma gratuita hasta 30/04/2024

## Stack tecnologico

En el stack tecnológico, hacemos uso de Python como nuestra herramienta principal, destacando las siguientes bibliotecas:

    pandas: Utilizada para la limpieza, modelado y análisis de datos.
    yfinance: Empleada para consumir la API y obtener información financiera.
    seaborn y matplotlib: Utilizadas para las visualizaciones.
    FastAPI: Implementado para montar una API eficiente que facilita la interacción con nuestro modelo.
    python-dotenv: Utilizado para proteger y gestionar información sensible, como credenciales para la base de datos.
    scikit-learn: Utilizado para la implementación y entrenamiento del modelo seleccionado.

Además, entre otras herramientas notables, destacamos Render, tanto para el despliegue y consumo de la API como para el montaje de la base de datos.

Este sólido stack tecnológico proporciona la base necesaria para el desarrollo, implementación y mantenimiento efectivo de nuestro proyecto de análisis financiero.

## Colaboración entre Equipos

### Data Engineering:
Un equipo dedicado a la ingeniería de datos proporciona una sólida base de datos y automatiza la carga de datos. La colaboración con este equipo garantiza la disponibilidad y actualización constante de los datos necesarios para el análisis financiero.

### Analistas de datos:
Nuestro equipo de analistas trabaja en un dashboard que facilita la presentación visual de los resultados y hallazgos. Este dashboard proporciona una interfaz intuitiva para interpretar y comunicar eficazmente las tendencias financieras descubiertas.
Repositorio General del Proyecto

Para acceder al repositorio completo del proyecto, donde trabajo el equipo completo: [Repositorio del equipo](https://github.com/No-Country/c16-96-m-data-bi).

## Contacto
Marcelo Peralta
| Forma de Contacto | Enlace                           |
|-------------------|----------------------------------|
| Correo            | [cheloperalta22@gmail.com](mailto:cheloperalta22@gmail.com)     |
| LinkedIn          | [Marcelo Peralta](https://www.linkedin.com/in/marcelo-peralta2) |
| WhatsApp          | [Mensaje por WhatsApp](https://wa.me/+5492616325753)  |

