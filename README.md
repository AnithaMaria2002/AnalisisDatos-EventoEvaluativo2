## Integrantes
- Juan David Gaviria Correa
- Ana Maria Valencia Restrepo
- Juan Sebastian Restrepo Nieto
- Juan Felipe Muñoz Rengifo

## Links Eventos Evaluativos

- [EventoEvaluativo2](https://github.com/AnithaMaria2002/AnalisisDatos-EventoEvaluativo2)
- [EventoEvaluativo3](https://github.com/JuanDa237/AnalisisDatos-EventoEvaluativo3)
- [EventoEvaluativo4](https://github.com/JuanDa237/EventoEvaluativo4-AnalisisDeDatos)
- [ProyectoFinal](https://github.com/JuanDa237/ProyectoFinal-AnalisisDeDatos)


# 📊 Análisis Avanzado de Datos: Predicción de Ingresos (Adult Income Dataset)

Este repositorio documenta el flujo completo de un proyecto de _Data Science_ enfocado en el **Adult Income Dataset** (Censo de Ingresos de EE. UU.). El objetivo principal es predecir si un individuo gana **más de $50,000 USD** al año, utilizando técnicas avanzadas de preprocesamiento, escalado robusto y **Análisis Discriminante Lineal (LDA)** para optimizar la dimensionalidad.

## 🚀 Estructura del Proyecto

El análisis se divide en tres fases principales, siguiendo la convención de _notebooks_ secuenciales, y utiliza el archivo `requirements.txt` para gestionar las dependencias.

| Archivo                               | Descripción                                                                                                                                                     | Fase del Análisis              |
| :------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------- |
| `01_exploracion_base_datos.ipynb`     | **Exploración Inicial.** Carga y estandarización de nombres de columnas. Primeros chequeos de tipos de datos y estructuras.                                     | **Exploración**                |
| `02_eda.ipynb`                        | **Análisis Exploratorio de Datos (EDA).** Manejo de valores faltantes (imputación/eliminación), análisis de _outliers_ y creación de la _feature_ `grupo_edad`. | **Limpieza y Entendimiento**   |
| `03_preprocesamiento_reduccion.ipynb` | **Preprocesamiento Avanzado y LDA.** Implementación del `ColumnTransformer`, escalado de variables y aplicación del LDA.                                        | **Preparación y Optimización** |
| `requirements.txt`                    | Lista de librerías esenciales de Python (`pandas`, `numpy`, `scikit-learn`, `seaborn`, etc.) para reproducir el entorno.                                        | **Configuración**              |
| `adult.csv`                           | El _dataset_ original del censo de ingresos de EE. UU.                                                                                                          | **Datos Fuente**               |
| `datos_limpios.csv`                   | El _dataset_ resultante después de la fase de EDA (`02_eda.ipynb`), listo para el preprocesamiento final.                                                       | **Datos Intermedios**          |

---

## 🛠️ Configuración y Ejecución

Para ejecutar el proyecto localmente, asegúrese de tener Python instalado y siga estos pasos:

1.  **Instalar Dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Ejecutar Notebooks:**
    Abra Jupyter Notebook o JupyterLab y ejecute los _notebooks_ en orden secuencial: `01_exploracion...`, `02_eda...`, y finalmente `03_preprocesamiento_reduccion.ipynb`.

---

### 1. Manejo de Variables Numéricas con Outliers (RobustScaler y Logaritmo)

Las variables con fuerte sesgo, como `ganancia_capital` y `perdidas_capital`, fueron tratadas en dos pasos cruciales:

- **Transformación Logarítmica:** Se aplicó la transformación **$\log(x+1)$** (usando `np.log1p`) para mitigar el sesgo y comprimir los valores atípicos extremos, que eran comunes en estas columnas.
- **Creación de _Dummies_:** Se crearon _features_ binarias (`tiene_ganancia_capital`, `tiene_perdida_capital`) para que el modelo pudiera capturar el valor informativo del cero (la mayoría de las observaciones).
- **Escalado Final:** Se utilizó el **`RobustScaler`** para escalar todas las variables numéricas finales. Este _scaler_ utiliza la Mediana y el Rango Intercuartílico (IQR), haciéndolo **inmune a los _outliers_** restantes.

### 2. Codificación Categórica

Las variables nominales como `clase_de_trabajo`, `ocupacion`, etc., fueron codificadas mediante **`OneHotEncoder`** dentro del `ColumnTransformer`, asegurando un tratamiento uniforme y sin fugas de datos.

---

## 🎯 Reducción de Dimensionalidad con LDA

En el _notebook_ `03`, se eligió el **Análisis Discriminante Lineal (LDA)** sobre el PCA, ya que el problema es de clasificación (`ingreso` binario).

- **Objetivo:** LDA es **supervisado** y está diseñado para encontrar el subespacio que **maximiza la separación entre las clases** de ingreso (0: `<=50K` y 1: `>50K`).
- **Resultado:** Con dos clases, LDA reduce la dimensionalidad a **una única Componente Discriminante (LD1)**. Esta componente representa el eje de máxima separación de las clases.
- **Poder Discriminatorio:** La varianza explicada por esta única componente representa su **poder discriminatorio**, que es la métrica clave.

### Visualización

La efectividad de LDA se visualiza mediante un **Gráfico de Densidad (KDE Plot)**, que superpone las distribuciones de ambas clases a lo largo de la Componente LD1. Una separación clara en este gráfico indica que la proyección de LDA ha sido exitosa para distinguir entre los grupos de altos y bajos ingresos.

**Autores:** Ana Maria Valencia Quintero, Juan David Gaviria Correa, Juan Sebastian Restrepo Nieto
