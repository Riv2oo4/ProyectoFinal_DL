import os
from glob import glob
import numpy as np

from ultralytics import YOLO

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


# =========================
# CONFIGURACIÓN BÁSICA
# =========================

DATA_CONFIG = "shelfvision.yaml"  # archivo YAML de dataset
BASE_MODEL = "yolov8n.pt"         # se puede cambiar a yolov8s.pt
RUNS_DIR = "runs_shelfvision"     # carpeta donde guarda resultados
RUN_NAME = "yolov8n_shelfvision3"  # nombre del experimento

TEST_IMAGES_DIR = os.path.join("data", "images", "test")
TEST_LABELS_DIR = os.path.join("data", "labels", "test")

CONF_THRESH = 0.3
IMG_SIZE = 640


# =========================
# 1. ENTRENAMIENTO
# =========================

def train_model(epochs: int = 50, batch: int = 8):
    """
    Entrena YOLOv8 en el dataset definido en shelfvision.yaml
    """
    print(">> Cargando modelo base:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print(">> Iniciando entrenamiento...")
    results = model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        workers=2,
        project=RUNS_DIR,
        name=RUN_NAME,
        patience=10,         # early stopping
    )

    print(">> Entrenamiento finalizado.")
    print(">> Resultados guardados en:", os.path.join(RUNS_DIR, "detect", RUN_NAME))
    return results


# =========================
# 2. EVALUACIÓN DETECCIÓN
# =========================

def evaluate_detection():
    """
    Evalúa el modelo en val/test y muestra métricas de detección (mAP, F1, etc.).
    """
    weights_path = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró {weights_path}. Entrena el modelo primero.")

    print(">> Cargando modelo entrenado:", weights_path)
    model = YOLO(weights_path)

    print(">> Evaluando detección...")
    metrics = model.val(
        data=DATA_CONFIG,
        imgsz=IMG_SIZE,
    )

    print("========== MÉTRICAS DE DETECCIÓN ==========")
    for k, v in metrics.results_dict.items():
        print(f"{k}: {v:.4f}")

    return metrics


# =========================
# 3. MAE DE CONTEO
# =========================

def contar_gt_por_imagen(labels_dir: str):
    """
    Lee todos los .txt de labels_dir y devuelve un dict:
    {nombre_imagen_sin_ext: conteo_objetos}
    """
    conteos = {}
    label_files = glob(os.path.join(labels_dir, "*.txt"))
    print(f">> Encontrados {len(label_files)} archivos de etiqueta en {labels_dir}")

    for lf in label_files:
        base = os.path.splitext(os.path.basename(lf))[0]
        with open(lf, "r") as f:
            lines = f.readlines()
        conteos[base] = len(lines)  # cada línea = 1 objeto
    return conteos


def evaluate_count_mae():
    """
    Calcula el MAE de conteo en el conjunto de test.
    """
    weights_path = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró {weights_path}. Entrena el modelo primero.")

    model = YOLO(weights_path)

    # Ground truth
    gt_counts = contar_gt_por_imagen(TEST_LABELS_DIR)

    # Imágenes de test
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob(os.path.join(TEST_IMAGES_DIR, ext)))


    if not image_files:
        raise FileNotFoundError(f"No se encontraron imágenes en {TEST_IMAGES_DIR}")

    print(f">> Encontradas {len(image_files)} imágenes de test.")

    # Predicciones
    pred_counts = {}  # {nombre_imagen_sin_ext: conteo_predicho}

    print(">> Ejecutando predicciones en test...")
    results = model.predict(
        source=image_files,
        conf=CONF_THRESH,
        imgsz=IMG_SIZE,
        verbose=False
    )

    for img_path, res in zip(image_files, results):
        base = os.path.splitext(os.path.basename(img_path))[0]
        num_det = 0
        if res.boxes is not None:
            num_det = len(res.boxes)
        pred_counts[base] = num_det

    # Emparejar GT y predicción
    y_true = []
    y_pred = []

    for base, gt in gt_counts.items():
        if base in pred_counts:
            y_true.append(gt)
            y_pred.append(pred_counts[base])

    if not y_true:
        raise RuntimeError("No se pudo emparejar GT con predicciones. Revisa nombres de archivos.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    print("========== MÉTRICA DE CONTEO ==========")
    print("MAE de conteo (facings por imagen):", mae)
    return mae, y_true, y_pred


# =========================
# 4. EXPLICABILIDAD (CAM)
# =========================

def explain_with_cam(image_path: str,
                     method: str = "EigenCAM",
                     out_dir: str = "xai_outputs"):
    """
    Genera un heatmap CAM (EigenCAM, GradCAM++, etc.)
    sobre una imagen usando best.pt.

    Requiere:
        pip install YOLOv8-Explainer
    """
    weights_path = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró {weights_path}. Entrena el modelo primero.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    os.makedirs(out_dir, exist_ok=True)

    # Import lazy para no obligar a tener el paquete en otros modos
    try:
        from YOLOv8_Explainer import yolov8_heatmap
    except ImportError:
        raise ImportError(
            "No tienes YOLOv8-Explainer instalado. "
            "Instala con: pip install YOLOv8-Explainer"
        )

    # Config siguiendo la docs del paquete
    cam_model = yolov8_heatmap(
        weight=weights_path,
        conf_threshold=CONF_THRESH,
        method=method,                  # "EigenCAM", "GradCAMPlusPlus", etc.
        layer=[10, 12, 14, 16, 18, -3], # capas sugeridas por el paquete
        ratio=0.02,
        show_box=True,
        renormalize=False,
    )

    # Ejecuta CAM
    images = cam_model(img_path=image_path)

    # Guarda outputs
    for i, im in enumerate(images):
        save_path = os.path.join(out_dir, f"cam_{method}_{i}.png")
        im.save(save_path)
        print("Heatmap guardado en:", save_path)

    return images


# =========================
# 5. DEMO CON GRADIO
# =========================

def launch_demo():
    """
    Lanza una interfaz Gradio para subir una imagen
    y ver detecciones + conteo.
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio no está instalado. Ejecuta: pip install gradio")

    weights_path = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró {weights_path}. Entrena el modelo primero.")

    model = YOLO(weights_path)
    names = model.names

    def detectar_y_contar(image):
        res = model(image, conf=CONF_THRESH, imgsz=IMG_SIZE)[0]
        plotted = res.plot()

        counts = {}
        if res.boxes is not None:
            for c in res.boxes.cls.cpu().numpy().tolist():
                c = int(c)
                counts[c] = counts.get(c, 0) + 1

        counts_legibles = {names[k]: v for k, v in counts.items()}
        total = sum(counts_legibles.values())

        texto = "Conteo por clase:\n"
        for nombre, num in counts_legibles.items():
            texto += f"- {nombre}: {num}\n"
        texto += f"\nTotal de facings: {total}"

        return plotted, texto

    demo = gr.Interface(
        fn=detectar_y_contar,
        inputs=gr.Image(type="numpy", label="Sube una imagen de estante"),
        outputs=[
            gr.Image(type="numpy", label="Detecciones"),
            gr.Textbox(label="Conteo")
        ],
        title="ShelfVision – Detección y Conteo en Estantes"
    )

    demo.launch()


# =========================
# MAIN 
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Proyecto ShelfVision")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval_det", "eval_count", "demo", "explain"],
                        help="Modo de ejecución")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas para entrenamiento")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")

    # argumentos para explain
    parser.add_argument("--img", type=str, default=None,
                        help="Ruta a una imagen para generar CAM")
    parser.add_argument("--method", type=str, default="EigenCAM",
                        choices=["EigenCAM","GradCAM","GradCAMPlusPlus","XGradCAM",
                                 "LayerCAM","EigenGradCAM","HiResCAM"],
                        help="Método CAM a usar")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(epochs=args.epochs, batch=args.batch)
    elif args.mode == "eval_det":
        evaluate_detection()
    elif args.mode == "eval_count":
        evaluate_count_mae()
    elif args.mode == "demo":
        launch_demo()
    elif args.mode == "explain":
        if args.img is None:
            raise ValueError("Debes pasar --img con la ruta de una imagen.")
        explain_with_cam(args.img, method=args.method)