import streamlit as st
from ultralytics import YOLO
import cv2
import PIL.Image
import numpy as np
import os
import requests
from io import BytesIO
from cap_from_youtube import cap_from_youtube

# 1. Configuración de la página
st.set_page_config(page_title="PPE Detector Pro", layout="wide")
st.title("🛡️ Detector de Seguridad Industrial")

# 2. Rutas del modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# 3. Carga del modelo
@st.cache_resource
def load_yolo():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

model = load_yolo()

if model is None:
    st.error(f"❌ No se encontró 'best.pt' en {BASE_DIR}")
    st.stop()

# --- INTERFAZ SIDEBAR ---
st.sidebar.title("Configuración")
conf_level = st.sidebar.slider("Confianza", 0.1, 1.0, 0.4)
source = st.sidebar.radio("Fuente", ["Imagen (Local/URL)", "Video (Local/YouTube)", "Cámara en Vivo"])

# --- FUNCIONES DE APOYO ---
def procesar_y_mostrar(frame):
    """Aplica detección y corrige color para Streamlit"""
    results = model.predict(frame, conf=conf_level, verbose=False)
    res_bgr = results[0].plot()
    # La clave para el color: YOLO bota BGR, Streamlit necesita RGB
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    return res_rgb

# --- LÓGICA DE DETECCIÓN ---

# MODO IMAGEN
if source == "Imagen (Local/URL)":
    sub_source = st.radio("Tipo de entrada", ["Subir Archivo", "Enlace de Imagen (URL)"])
    img_input = None

    if sub_source == "Subir Archivo":
        file = st.file_uploader("Subir imagen", type=['jpg', 'jpeg', 'png'])
        if file:
            img_input = PIL.Image.open(file)
    else:
        url = st.text_input("Pega el link de la imagen:")
        if url:
            try:
                response = requests.get(url)
                img_input = PIL.Image.open(BytesIO(response.content))
            except:
                st.error("No se pudo cargar la imagen desde ese link.")

    if img_input:
        # Convertimos a formato OpenCV (BGR) para procesar
        open_cv_image = np.array(img_input.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy() # Convertimos RGB a BGR
        
        resultado = procesar_y_mostrar(open_cv_image)
        st.image(resultado, caption="Detección Finalizada", use_container_width=True)

# MODO VIDEO
elif source == "Video (Local/YouTube)":
    sub_source = st.radio("Tipo de entrada", ["Subir Archivo", "YouTube Link"])
    cap = None

    if sub_source == "Subir Archivo":
        vid_file = st.file_uploader("Subir video", type=['mp4', 'mov'])
        if vid_file:
            with open("temp.mp4", "wb") as f:
                f.write(vid_file.read())
            cap = cv2.VideoCapture("temp.mp4")
    else:
        yt_url = st.text_input("Pega el link de YouTube:")
        if yt_url:
            try:
                cap = cap_from_youtube(yt_url, '720p')
            except Exception as e:
                st.error(f"Error al cargar video de YouTube: {e}")

    if cap:
        frame_placeholder = st.empty()
        stop_btn = st.button("Detener Proceso")
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret: break
            
            resultado = procesar_y_mostrar(frame)
            frame_placeholder.image(resultado, channels="RGB")
        cap.release()

# MODO CÁMARA
elif source == "Cámara en Vivo":
    run = st.checkbox('Iniciar Cámara de mi PC')
    if run:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al acceder a la cámara.")
                break
            
            resultado = procesar_y_mostrar(frame)
            frame_placeholder.image(resultado, channels="RGB")
        cap.release()