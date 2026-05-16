# Dashboard ERC Araucanía

Dashboard interactivo para el análisis de prevalencia de ERC en la Región de la Araucanía.

## Instalación

```bash
# 1. Crear entorno (recomendado)
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Cómo correr

Coloca `CKD_DATA_v2.xlsx` en la misma carpeta que `app.py` y ejecuta:

```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador (http://localhost:8501).

Si prefieres subir el archivo desde la app, simplemente corre `streamlit run app.py` sin el xlsx y usa el uploader del sidebar.

## Filtros disponibles

**En el sidebar (afectan a todas las pestañas):**
- Rango de edad (slider)
- Comunas (multiselect, default: todas)
- Pueblo originario (PPOO / No PPOO)
- Sexo (Mujer / Hombre)
- Estado nutricional (bajo peso / peso normal / sobrepeso / obeso)
- HTA autoreportada (todos / solo conocida / solo sin conocer)
- DM autoreportada (todos / solo conocida / solo sin conocer)

## Pestañas

1. **🏠 Resumen** — Características generales, prevalencias principales, pirámide etaria, distribución eGFR
2. **📊 Prevalencia** — Prevalencias por definición, estadios KDIGO, matriz eGFR×Albuminuria
3. **🔬 Estratificación** — Prevalencia por variable a elegir + comparación PPOO×edad
4. **👥 Subgrupos** — Carga acumulada de factores, combinaciones clínicas
5. **⚠️ Factores de riesgo** — RR crudo (forest plot) y regresión logística multivariada
6. **📈 Score de riesgo** — Performance del score, ROC, calculadora individual
7. **🔄 Concordancia** — eGFR 1ra vs 2da toma, Bland-Altman, reclasificación
8. **📋 Datos** — Vista tabular y descarga de la muestra filtrada como CSV

## Notas

- Los cálculos se actualizan automáticamente al cambiar los filtros.
- Los IC95% se calculan con método Wilson para proporciones.
- La regresión logística se reajusta sobre la muestra filtrada; si N es insuficiente la pestaña avisa.
- Todos los gráficos son interactivos (zoom, hover, descarga PNG desde el menú del gráfico).
