# 📚 ¿Dónde publicar mi investigación? — Scopus + SCImago + IA gratuita (OpenRouter)

App en **Streamlit** que recibe el tema de tu investigación (p. ej. *"relación entre las conductas de
ciudadanía organizacional y el compromiso organizacional de los trabajadores de una empresa"*) y devuelve las
revistas **indexadas en Scopus y SCImago** (áreas Administración y Psicología, configurables) donde mejor encaja,
con cuartil **Q1–Q4**, SJR, ISSN, eISSN, editorial, país, enlaces y si es **gratis o de pago** para publicar.

## Cómo funciona

1. Escribes el tema y pegas tu **API key de OpenRouter** (gratis en <https://openrouter.ai/keys>).
2. **Paso 1 (IA):** un LLM gratuito traduce/interpreta el tema y elige las categorías SCImago y palabras clave.
3. **Filtro local:** el programa filtra las ~3 000 revistas del área, puntúa por categorías + título + SJR y
   preselecciona las mejores candidatas.
4. **Paso 2 (IA):** el LLM analiza las candidatas, elige las más afines y justifica cada una.
   Los IDs que el modelo invente se descartan: solo se muestran revistas reales de tus bases.
5. Resultados con filtros *Gratis / De pago*, detalle por revista y descarga en **Excel**.

**Fallback automático:** los modelos se prueban en orden (por defecto `openrouter/free`, Flint, Nemotron 3.5 Super VL,
Nemotron 3 Ultra, Nemotron 3 Nano Omni, Ling 3.2 Flash…). Si uno da error (límite 429, caída, respuesta inválida)
pasa al siguiente. La lista se actualiza en vivo desde OpenRouter y puedes agregar cualquier ID `…:free`.

## ¿Gratis o de pago? (importante)

Ni Scopus ni SCImago publican el monto de las tasas (APC). La app lo **deduce** de SCImago:

| Etiqueta | Criterio | Significado |
|---|---|---|
| **GRATIS** | Open Access *Diamond* | Sin costo para autor ni lector |
| **PAGO (APC probable)** | Open Access (no Diamond) | Las revistas OA suelen cobrar al autor |
| **Normalmente GRATIS (suscripción)** | No Open Access | Se paga por leer; publicar suele ser gratis (puede haber opción híbrida de pago) |
| **Verificar** | Sin dato | Revista sin métricas SCImago |

Confirma siempre en el sitio de la revista (la app incluye enlaces a SCImago, Scopus, DOAJ y búsqueda de
guía para autores). Las revistas no tienen DOI propio (los DOI son de artículos); se muestran ISSN y eISSN.

## Estructura

```
app.py                         # aplicación Streamlit
build_dataset.py               # genera la base a partir de tus xlsx/csv originales
requirements.txt
data/revistas_procesadas.csv.gz  # Scopus (Mar 2026) + SCImago 2025 ya cruzados (48 360 fuentes)
Revistas_Colab.ipynb           # ejecutar la app desde Google Colab
```

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Regenerar la base con archivos más nuevos

```bash
python build_dataset.py ext_list_Mar_2026.xlsx scimagojr_2025.csv
```
(cruza por `Sourceid`/`Sourcerecord ID`; usa la primera hoja del Excel de Scopus). También puedes borrar
`data/revistas_procesadas.csv.gz` y la app te pedirá subir los dos archivos.

## Subir a GitHub

```bash
git init
git add .
git commit -m "App buscador de revistas Scopus + SCImago"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```
El archivo procesado pesa ~3.7 MB, muy por debajo del límite de GitHub.

## Desplegar en Streamlit Community Cloud

1. Entra a <https://share.streamlit.io> con tu cuenta de GitHub → **Create app**.
2. Elige tu repositorio, rama `main` y archivo principal **`app.py`**. → **Deploy**.
3. Cada usuario ingresa **su propia API key** en la barra lateral (no se guarda).
   Opcional: define `OPENROUTER_API_KEY` en *Settings → Secrets* para usar una clave por defecto.

## Google Colab

Abre `Revistas_Colab.ipynb` en Colab, ejecuta las celdas y entra por el enlace público que genera
(túnel Cloudflare).

## Fuentes de datos

- Scopus Source List, marzo 2026 (Elsevier).
- SCImago Journal & Country Rank 2025 (<https://www.scimagojr.com>).
- Modelos gratuitos: <https://openrouter.ai/collections/free-models> (los modelos gratuitos tienen límites de uso y cambian con frecuencia).
