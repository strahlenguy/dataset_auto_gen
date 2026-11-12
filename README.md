# sam3-dataset

Herramientas para generar datasets de recortes de objetos usando **SAM 3**.
Le das un directorio de imágenes y un prompt de texto, y te regresa los objetos
recortados, organizados por clase y listos para entrenar.

Material del curso de **Visión Artificial** (CUTLAJO).

## Estructura

```
sam3-dataset/
├── scripts/
│   ├── descargar_pesos_sam3.py     # baja y verifica los pesos
│   ├── sam3_recortar_dataset.py    # segmenta y recorta
│   └── empaquetar_roboflow.py      # convierte la salida en zips para Roboflow / YOLO
├── pesos/                          # sam3.pt va aquí (ignorado por git)
├── datos/
│   ├── raw/                        # imágenes de entrada (ignorado)
│   ├── referencias/                # fotos del objeto buscado (ignorado, opcional)
│   └── salida/                     # el dataset generado (ignorado)
├── requirements.txt
└── README.md
```

Las cuatro carpetas de datos y pesos están en `.gitignore`. El repo lleva solo
código: los pesos pesan GB y tienen su propia licencia, y las imágenes son de
cada quien. Los `.gitkeep` existen para que la estructura sí se clone.

## Instalación

```bash
git clone <url-del-repo>
cd sam3-dataset

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install git+https://github.com/facebookresearch/sam3.git
pip install -r requirements.txt
```

Requiere Python >= 3.12 y GPU con CUDA. En CPU funciona, pero es muy lento.

## Pesos del modelo

1. Pide acceso en https://huggingface.co/facebook/sam3 y espera la aprobación.
2. Autentícate: `hf auth login`
3. Descarga:

```bash
python scripts/descargar_pesos_sam3.py -d pesos/
```

Si alguien te compartió los pesos por un espejo interno (sin necesidad de
cuenta ni token):

```bash
python scripts/descargar_pesos_sam3.py -d pesos/ \
    --url http://<ip-del-servidor>:8000/sam3.pt --sha256 <hash>
```

El uso de los pesos está sujeto a la SAM License. Si los redistribuyes, tienes
que incluir una copia de esa licencia; el script la deja junto al `.pt`.

## Uso

```bash
# 1. Pon tus imágenes en datos/raw/

# 2. Calibra umbrales con pocas imágenes
python scripts/sam3_recortar_dataset.py \
    -i datos/raw -o datos/salida/prueba \
    -p "casco=hard hat" \
    --checkpoint pesos/sam3.pt --limite 20

# 3. Revisa datos/salida/prueba/mosaico_*.jpg y metadatos.csv,
#    ajusta --umbral y --area-min, y corre todo

python scripts/sam3_recortar_dataset.py \
    -i datos/raw -o datos/salida/v1 \
    -p "casco=hard hat" -p "chaleco=safety vest" \
    --checkpoint pesos/sam3.pt \
    --fondo transparente --tamano 224 --exportar-yolo
```

Los prompts van en inglés, como frases nominales cortas. El formato
`clase=frase` deja la carpeta en español y el prompt en el idioma del modelo.

Ayuda completa de cada script:

```bash
python scripts/sam3_recortar_dataset.py --help
python scripts/descargar_pesos_sam3.py --help
```

## Salida

```
datos/salida/v1/
├── recortes/
│   ├── casco/            # un archivo por objeto
│   └── chaleco/
├── metadatos.csv         # score, caja, área y similitud de cada recorte
├── mosaico_casco.jpg     # hoja de contacto para revisar de un vistazo
├── resumen.json          # parámetros usados en la corrida
└── yolo/                 # solo con --exportar-yolo (polígonos de segmentación, sin dividir)
    ├── images/
    ├── labels/
    └── data.yaml
```

Revisa siempre los mosaicos antes de entrenar. SAM 3 etiqueta muy bien, pero no
distingue variantes específicas de un objeto, así que el filtro humano sigue
siendo parte del proceso.

## Empaquetar para Roboflow (detección con cajas)

`--exportar-yolo` deja polígonos de **segmentación** en una sola carpeta. Para
un proyecto de **Object Detection** en Roboflow o con `ultralytics`, convierte
la corrida a cajas, divídela en train/valid/test y empaquétala en zip:

```bash
python scripts/empaquetar_roboflow.py datos/salida/v1 \
    --score-min 0.3 --muestras 150 500 --sin-detectar --vista-previa
```

Lee `metadatos.csv` y `resumen.json`, así que no repites clases ni rutas (si
moviste las imágenes originales, indícalas con `--entrada`). Además:

- Quita **fragmentos**: cajas que caen casi completas (`--contenida 0.8`)
  dentro de otra mayor de la misma clase, típicas cuando una máscara sale en
  varios pedazos.
- Convierte BMP/TIFF/WebP a JPG y aplica la rotación EXIF, para que las cajas
  coincidan con la imagen que ve Roboflow.
- Divide 70/20/10 por defecto (`--division 0.8 0.1 0.1` para cambiarlo).

```
datos/salida/v1/roboflow/
├── v1_yolo.zip                  # todo: train/ valid/ test/ + data.yaml
├── v1_muestra150_yolo.zip       # con --muestras: imágenes de score alto
├── v1_muestra500_yolo.zip
├── sin_detectar.zip             # con --sin-detectar: para etiquetar a mano
├── revisar_confianza_baja.csv   # imágenes con alguna caja de score < 0.5
├── vista_previa_cajas.jpg       # con --vista-previa
└── resumen_empaquetado.json
```

**Subir a Roboflow:** Create New Project → Object Detection → Upload → arrastra
el zip **sin descomprimir**. Roboflow lee el `data.yaml`, carga las cajas y
conserva la división. Revisa las imágenes de `revisar_confianza_baja.csv` en el
anotador antes de generar la versión.

**Entrenar directo con ultralytics:** descomprime el zip y cambia el
`data.yaml` a rutas absolutas:

```yaml
path: /ruta/a/v1_yolo
train: train/images
val: valid/images
test: test/images
names: ['casco', 'chaleco']   # respeta el orden original
```

## Licencia

El código de este repo: elige la que quieras (MIT, Apache 2.0) y agrega el
archivo `LICENSE`. Los pesos de SAM 3 **no** están cubiertos por ella; se rigen
por la SAM License de Meta.
