![imagen](https://github.com/Tato2013/ProyectoNoCountryML/assets/101934772/018322db-192d-4c44-8725-35c0dab4844b)

# Análisis de Tendencias Financieras en Acciones




## Introducción 

Bienvenido al proyecto de simulación de consultoría financiera GROWTREND, donde llevamos a cabo análisis profundos para guiar futuras inversiones con enfoque técnico. Nuestro objetivo es proporcionar orientación estratégica a los clientes, sugiriendo inversiones en acciones en momentos óptimos para maximizar ganancias a corto y largo plazo.

En el dinámico escenario financiero actual, la toma de decisiones informada es esencial. A través de este proyecto, hemos desarrollado un modelo de Machine Learning que se basa en el estudio de datos de acciones de los últimos 5 años. Este modelo nos permite comprender posibles tendencias futuras, ofreciendo a nuestros clientes una herramienta valiosa para la toma de decisiones financieras.

Exploraremos la metodología específica y los algoritmos de Machine Learning utilizados en este análisis, proporcionando una visión detallada de cómo abordamos la complejidad del mercado financiero. Creemos que la simulación y el análisis de datos históricos son fundamentales para anticipar cambios y brindar a nuestros clientes la ventaja competitiva necesaria en el mundo de las inversiones.

GROWTREND no solo busca ofrecer información, sino también proporcionar beneficios tangibles a nuestros clientes. Nuestra simulación no solo revela posibles tendencias, sino que también respalda decisiones informadas que pueden influir positivamente en el rendimiento de las inversiones.

Únete a nosotros en este viaje hacia el conocimiento financiero avanzado. Con GROWTREND, transformamos datos en estrategias y convertimos la incertidumbre en oportunidad.

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
Aca pueden ver su trabajo realizado [Dashboard](https://www.novypro.com/project/dashboard-de-análisis-de-tendencias-de-acciones)
Repositorio General del Proyecto

Para acceder al repositorio completo del proyecto, donde trabajo el equipo completo: [Repositorio del equipo](https://github.com/No-Country/c16-96-m-data-bi).

## Contacto
Marcelo Peralta
| Forma de Contacto | Enlace                           |
|-------------------|----------------------------------|
| Correo            | [cheloperalta22@gmail.com](mailto:cheloperalta22@gmail.com)     |
| LinkedIn          | [Marcelo Peralta](https://www.linkedin.com/in/marcelo-peralta2) |
| WhatsApp          | [Mensaje por WhatsApp](https://wa.me/+5492616325753)  |

