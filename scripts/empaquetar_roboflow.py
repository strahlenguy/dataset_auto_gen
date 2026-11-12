#!/usr/bin/env python3
"""
Convierte la salida de sam3_recortar_dataset.py en un dataset de DETECCIÓN
(cajas) en formato YOLO, dividido en train/valid/test y empaquetado en zip,
listo para arrastrarlo a Roboflow o entrenar con ultralytics.

Lee metadatos.csv y resumen.json de la corrida, así que no hay que repetir
las clases ni la carpeta de imágenes.

Qué hace, además de convertir:
  - Quita "fragmentos": cajas que caen casi completas dentro de otra caja más
    grande de la misma clase (p. ej. un pedazo de la máscara que salió aparte).
  - Convierte BMP/TIFF/WebP a JPG y aplica la orientación EXIF, para que las
    cajas coincidan con lo que ve Roboflow.
  - Opcional: zips de muestra con las imágenes de mayor confianza, un zip con
    las imágenes donde SAM 3 no encontró nada (para etiquetar a mano), una
    lista de imágenes a revisar y una hoja de vista previa con las cajas.

Uso típico:

    python scripts/empaquetar_roboflow.py datos/salida/v1

    python scripts/empaquetar_roboflow.py datos/salida/v1 \\
        --score-min 0.3 --muestras 150 500 --sin-detectar --vista-previa

Resultado (por defecto en datos/salida/v1/roboflow/):

    <nombre>_yolo.zip               train/ valid/ test/ + data.yaml
    <nombre>_muestra150_yolo.zip    (con --muestras)
    sin_detectar.zip                (con --sin-detectar)
    revisar_confianza_baja.csv      imágenes con alguna caja de score bajo
    vista_previa_cajas.jpg          (con --vista-previa)
    resumen_empaquetado.json

En Roboflow: Create Project → Object Detection → Upload → arrastra el zip
(sin descomprimir). Roboflow lee el data.yaml y conserva la división.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SE_COPIAN_TAL_CUAL = {".jpg", ".jpeg", ".png"}
COLORES = [(255, 0, 0), (0, 200, 255), (0, 255, 0), (255, 200, 0), (255, 0, 255), (255, 128, 0)]


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def interseccion(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def nombre_plano(rel: str) -> str:
    """'sub/carpeta/foto.jpg' -> 'sub__carpeta__foto.jpg' (igual que el script de SAM 3)."""
    return rel.replace("/", "__")


def listar_imagenes(directorio: Path, recursivo: bool) -> list[Path]:
    patron = "**/*" if recursivo else "*"
    return sorted(p for p in directorio.glob(patron) if p.is_file() and p.suffix.lower() in EXTENSIONES)


def guardar_imagen(origen: Path, destino_sin_ext: Path) -> tuple[Path, int, int]:
    """Copia la imagen al dataset. Re-codifica a JPG si el formato no es JPG/PNG
    o si trae rotación EXIF (las cajas de SAM 3 están en la imagen ya rotada)."""
    img = Image.open(origen)
    orientacion = img.getexif().get(0x0112, 1)
    ext = origen.suffix.lower()
    if ext in SE_COPIAN_TAL_CUAL and orientacion == 1:
        destino = destino_sin_ext.with_suffix(ext)
        shutil.copy2(origen, destino)
        return destino, img.width, img.height
    img = ImageOps.exif_transpose(img).convert("RGB")
    destino = destino_sin_ext.with_suffix(".jpg")
    img.save(destino, quality=95)
    return destino, img.width, img.height


# --------------------------------------------------------------------------- #
# Lectura de la corrida de SAM 3
# --------------------------------------------------------------------------- #
def leer_corrida(salida: Path, entrada_cli: Path | None):
    resumen_path = salida / "resumen.json"
    meta_path = salida / "metadatos.csv"
    if not meta_path.exists():
        sys.exit(f"No encuentro {meta_path}. ¿Es la carpeta de salida de sam3_recortar_dataset.py?")
    resumen = json.loads(resumen_path.read_text(encoding="utf-8")) if resumen_path.exists() else {}
    params = resumen.get("parametros", {})

    clases = list(resumen.get("prompts", {}).keys())
    entrada = entrada_cli or (Path(params["entrada"]) if params.get("entrada") else None)
    if entrada is None or not entrada.exists():
        sys.exit(f"No encuentro la carpeta de imágenes originales ({entrada}). Indícala con --entrada.")

    filas = list(csv.DictReader(open(meta_path, encoding="utf-8")))
    for f in filas:  # clases que no estaban en el resumen (por si acaso)
        if f["clase"] not in clases:
            clases.append(f["clase"])
    return entrada, bool(params.get("recursivo", False)), clases, filas


def construir_cajas(filas, score_min: float, contenida: float):
    """{imagen_rel: [(clase, score, (x0,y0,x1,y1)), ...]} sin fragmentos."""
    por_img = collections.defaultdict(list)
    for f in filas:
        if f.get("aceptado", "1") != "1":
            continue
        s = float(f["score"])
        if s < score_min:
            continue
        caja = tuple(int(float(f[k])) for k in ("x0", "y0", "x1", "y1"))
        por_img[f["imagen_origen"]].append((f["clase"], s, caja))

    limpias, fragmentos = {}, 0
    for img, dets in por_img.items():
        dets.sort(key=lambda d: -area(d[2]))  # grandes primero
        conservadas = []
        for clase, s, caja in dets:
            es_fragmento = any(
                c == clase and interseccion(caja, k) / max(area(caja), 1) > contenida
                for c, _, k in conservadas
            )
            if es_fragmento:
                fragmentos += 1
                continue
            conservadas.append((clase, s, caja))
        limpias[img] = conservadas
    return limpias, fragmentos


# --------------------------------------------------------------------------- #
# Empaquetado
# --------------------------------------------------------------------------- #
def lineas_yolo(dets, clases, ancho, alto):
    lineas = []
    for clase, _, (x0, y0, x1, y1) in dets:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(ancho, x1), min(alto, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2 / ancho, (y0 + y1) / 2 / alto
        w, h = (x1 - x0) / ancho, (y1 - y0) / alto
        lineas.append(f"{clases.index(clase)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lineas


def armar_zip(nombre, imagenes, cajas, entrada, clases, division, destino, rng):
    imagenes = list(imagenes)
    rng.shuffle(imagenes)
    n = len(imagenes)
    c1 = round(n * division[0])
    c2 = c1 + round(n * division[1])
    partes = {"train": imagenes[:c1], "valid": imagenes[c1:c2], "test": imagenes[c2:]}

    tmp = destino / f".{nombre}_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    for parte, lista in partes.items():
        (tmp / parte / "images").mkdir(parents=True)
        (tmp / parte / "labels").mkdir(parents=True)
        for rel in lista:
            base = Path(nombre_plano(rel)).stem
            img_dest, w, h = guardar_imagen(entrada / rel, tmp / parte / "images" / base)
            (tmp / parte / "labels" / f"{img_dest.stem}.txt").write_text(
                "\n".join(lineas_yolo(cajas[rel], clases, w, h)) + "\n", encoding="utf-8")

    yaml = ["train: ../train/images", "val: ../valid/images", "test: ../test/images", "",
            f"nc: {len(clases)}", "names: [" + ", ".join(f"'{c}'" for c in clases) + "]"]
    (tmp / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    zip_path = destino / f"{nombre}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)
    conteo = {k: len(v) for k, v in partes.items()}
    print(f"  {zip_path.name}: {n} imágenes {conteo}, {zip_path.stat().st_size / 1e6:.1f} MB")
    return {"zip": zip_path.name, "imagenes": n, **conteo}


def vista_previa(cajas, entrada, clases, destino, rng, n=48, lado=200, columnas=8):
    sel = rng.sample(sorted(cajas), min(n, len(cajas)))
    if not sel:
        return
    filas = (len(sel) + columnas - 1) // columnas
    hoja = Image.new("RGB", (columnas * lado, filas * lado), (30, 30, 30))
    for i, rel in enumerate(sel):
        im = ImageOps.exif_transpose(Image.open(entrada / rel)).convert("RGB")
        d = ImageDraw.Draw(im)
        for clase, _, caja in cajas[rel]:
            d.rectangle(caja, outline=COLORES[clases.index(clase) % len(COLORES)], width=max(2, im.width // 90))
        im.thumbnail((lado, lado))
        hoja.paste(im, ((i % columnas) * lado + (lado - im.width) // 2, (i // columnas) * lado + (lado - im.height) // 2))
    hoja.save(destino / "vista_previa_cajas.jpg", quality=88)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Empaqueta la salida de SAM 3 como dataset de detección YOLO para Roboflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("salida_sam3", type=Path, help="Carpeta de salida de sam3_recortar_dataset.py (con metadatos.csv)")
    p.add_argument("-o", "--destino", type=Path, default=None, help="Dónde dejar los zips (por defecto <salida_sam3>/roboflow)")
    p.add_argument("--entrada", type=Path, default=None, help="Carpeta de imágenes originales (por defecto, la de resumen.json)")
    p.add_argument("--nombre", default=None, help="Nombre base de los zips (por defecto, el de la carpeta de salida)")
    p.add_argument("--score-min", type=float, default=0.0, help="Descartar cajas con score menor")
    p.add_argument("--contenida", type=float, default=0.8,
                   help="Fracción de una caja dentro de otra mayor (misma clase) para considerarla fragmento; 1.0 = no filtrar")
    p.add_argument("--division", type=float, nargs=3, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VALID", "TEST"))
    p.add_argument("--muestras", type=int, nargs="*", default=[], help="Tamaños de zips de muestra extra, p. ej. 150 500")
    p.add_argument("--score-muestra", type=float, default=0.5, help="Score mínimo de todas las cajas de una imagen para entrar a las muestras")
    p.add_argument("--umbral-revisar", type=float, default=0.5, help="Imágenes con alguna caja por debajo van a revisar_confianza_baja.csv")
    p.add_argument("--sin-detectar", action="store_true", help="Zip aparte con las imágenes sin ninguna caja")
    p.add_argument("--vista-previa", action="store_true", help="Hoja con 48 imágenes al azar y sus cajas")
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    if abs(sum(args.division) - 1) > 1e-6:
        sys.exit("--division debe sumar 1 (p. ej. 0.7 0.2 0.1)")

    entrada, recursivo, clases, filas = leer_corrida(args.salida_sam3, args.entrada)
    cajas, fragmentos = construir_cajas(filas, args.score_min, args.contenida)
    cajas = {k: v for k, v in cajas.items() if v}
    if not cajas:
        sys.exit("No quedó ninguna caja. Baja --score-min o revisa la corrida de SAM 3.")

    destino = args.destino or (args.salida_sam3 / "roboflow")
    destino.mkdir(parents=True, exist_ok=True)
    nombre = args.nombre or args.salida_sam3.resolve().name
    rng = random.Random(args.semilla)

    todas = [p.relative_to(entrada).as_posix() for p in listar_imagenes(entrada, recursivo)]
    sin_det = [r for r in todas if r not in cajas]
    por_img = collections.Counter(len(v) for v in cajas.values())
    por_clase = collections.Counter(c for v in cajas.values() for c, _, _ in v)

    print(f"Imágenes originales: {len(todas)}  con cajas: {len(cajas)}  sin detectar: {len(sin_det)}")
    print(f"Cajas: {sum(por_clase.values())} {dict(por_clase)}  fragmentos quitados: {fragmentos}")
    print("Zips:")

    resumen = {"entrada": str(entrada), "clases": clases, "score_min": args.score_min,
               "contenida": args.contenida, "division": args.division,
               "imagenes_originales": len(todas), "imagenes_con_cajas": len(cajas),
               "sin_detectar": len(sin_det), "cajas_por_clase": dict(por_clase),
               "cajas_por_imagen": dict(sorted(por_img.items())), "fragmentos_quitados": fragmentos, "zips": []}

    resumen["zips"].append(armar_zip(f"{nombre}_yolo", cajas.keys(), cajas, entrada, clases, args.division, destino, rng))

    score_min_img = {r: min(s for _, s, _ in v) for r, v in cajas.items()}
    seguras = sorted(r for r, s in score_min_img.items() if s >= args.score_muestra)
    for n in args.muestras:
        if n > len(seguras):
            print(f"  (muestra de {n} omitida: solo hay {len(seguras)} imágenes con score >= {args.score_muestra})")
            continue
        resumen["zips"].append(armar_zip(f"{nombre}_muestra{n}_yolo", rng.sample(seguras, n), cajas,
                                         entrada, clases, args.division, destino, rng))

    if args.sin_detectar and sin_det:
        zp = destino / "sin_detectar.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for r in sin_det:
                z.write(entrada / r, nombre_plano(r))
        print(f"  {zp.name}: {len(sin_det)} imágenes sin etiquetas, {zp.stat().st_size / 1e6:.1f} MB")
        resumen["zips"].append({"zip": zp.name, "imagenes": len(sin_det)})

    revisar = sorted(((r, s) for r, s in score_min_img.items() if s < args.umbral_revisar), key=lambda x: x[1])
    with open(destino / "revisar_confianza_baja.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "n_cajas", "score_min"])
        for r, s in revisar:
            w.writerow([r, len(cajas[r]), f"{s:.3f}"])
    print(f"  revisar_confianza_baja.csv: {len(revisar)} imágenes con alguna caja < {args.umbral_revisar}")

    if args.vista_previa:
        vista_previa(cajas, entrada, clases, destino, rng)
        print("  vista_previa_cajas.jpg")

    (destino / "resumen_empaquetado.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo: {destino.resolve()}")


if __name__ == "__main__":
    main()
