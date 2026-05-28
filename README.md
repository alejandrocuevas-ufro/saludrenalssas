# Dashboard ERC Araucanía

Dashboard interactivo en Streamlit para explorar la prevalencia de enfermedad renal crónica (ERC) en una muestra de pesquisa comunitaria de la Región de La Araucanía, Chile.

La aplicación utiliza eGFR CKD-EPI 2021, planillas curadas/estandarizadas y visualizaciones interactivas para describir prevalencia, estratificación, carga acumulada de factores de riesgo, asociaciones crudas/ajustadas y concordancia entre definiciones.

## Archivos principales

La estructura recomendada del repositorio es:

```text
.
├── app_v4.2.13.py
├── requirements.txt
├── README.md
├── CKD_DATA_v3_3.xlsx
├── 01 - Variable ¿Cual otra condición de salud?.xlsx
├── 02 - Variable Otros Antec Fliares OK.xlsx
├── 03 - consumo_AINEs_estandarizado_variables_modelo.xlsx
├── 04 - otros_diagnosticos_estandarizados_variables_modelo ok.xlsx
└── 05 - evaluacion_medica_estandarizada_variables_modelo.xlsx
```

También puedes renombrar `app_v4.2.13.py` a `app.py` si prefieres un nombre más simple para Streamlit.

## Archivos de datos esperados

La app puede cargar archivos de dos formas:

1. **Carga automática** desde la carpeta del repositorio.
2. **Carga manual** desde el panel lateral de Streamlit.

Para la carga automática, la app busca la base principal con uno de estos nombres:

```text
CKD_DATA_v3_3.xlsx
CKD_DATA_v3.3.xlsx
```

Las planillas complementarias 01–05 se detectan mediante patrones de nombre. Se recomienda mantener nombres que comiencen con `01`, `02`, `03`, `04` y `05`, respectivamente.

## Instalación local

Crea un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución local

Si el archivo principal se mantiene como `app_v4.2.13.py`:

```bash
streamlit run app_v4.2.13.py
```

Si lo renombras como `app.py`:

```bash
streamlit run app.py
```

## Publicación en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Verifica que `requirements.txt` esté en la raíz del repositorio.
3. En Streamlit Community Cloud, selecciona:
   - Repository: el repositorio de GitHub.
   - Branch: la rama principal, por ejemplo `main`.
   - Main file path: `app_v4.2.13.py` o `app.py`, según el nombre usado.
4. Selecciona una versión de Python compatible, idealmente Python 3.11 o 3.12.
5. Despliega la aplicación.

## Dependencias

Las dependencias principales son:

- Streamlit
- pandas
- NumPy
- Plotly
- SciPy
- statsmodels
- scikit-learn
- openpyxl

`openpyxl` es necesario para que pandas pueda leer archivos `.xlsx`.

## Consideraciones de privacidad

Este proyecto trabaja con información de salud. Antes de subir planillas a GitHub o publicar la app, confirma que los datos estén anonimizados y que el repositorio tenga el nivel de privacidad adecuado.

Si las planillas contienen datos sensibles o potencialmente identificables, usa un repositorio privado y evita publicar datos personales, identificadores directos o información clínica individual no anonimizada.

## Nota metodológica

La app calcula prevalencias con intervalos de confianza, estratificación por variables clínicas/demográficas, carga acumulada de factores de riesgo y razones de prevalencia crudas o ajustadas mediante modelos de Poisson con varianza robusta cuando corresponde.

Los resultados son descriptivos/exploratorios y deben interpretarse considerando el diseño transversal de la muestra de pesquisa.
