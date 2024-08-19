import streamlit as st
import numpy as np
import tensorflow as tf
import pydicom as dicom
import cv2
from PIL import Image
import os
from keras import backend as K
from fpdf import FPDF
import tempfile
from io import BytesIO

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Ruta del modelo
model_path = "clasificador_fractura.h5"

# Cargar el modelo
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Modelo cargado exitosamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Función para preprocesar la imagen
def preprocess(array):
    array = cv2.resize(array, (128, 128))  # Redimensionar a (128, 128)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # Asegurar que sea RGB
    array = array / 255.0  # Normalizar los valores de píxel entre 0 y 1
    array = np.expand_dims(array, axis=0)  # Añadir dimensión de lote (batch)
    return array

def grad_cam(array):
    img = preprocess(array)
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]

    last_conv_layer = model.get_layer("conv2d_2")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)

    for filters in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Redimensionar heatmap a las dimensiones de la imagen original
    heatmap = cv2.resize(heatmap, (array.shape[1], array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if len(array.shape) == 2 or array.shape[2] == 1:
        img2 = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        img2 = array

    img2 = cv2.resize(img2, (heatmap.shape[1], heatmap.shape[0]))   
    hif = 0.4 
    transparency = cv2.addWeighted(img2, 1 - hif, heatmap, hif, 0)
    
    return transparency[:, :, ::-1]

# Función de predicción actualizada con los nuevos labels
def predict(array):
    batch_array_img = preprocess(array)
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    
    # Diccionario con los labels
    labels = {
        0: 'Codo',
        1: 'Dedo',
        2: 'Antebrazo',
        3: 'Mano',
        4: 'Húmero',
        5: 'Hombro',
        6: 'Muñeca'
    }

    # Obtener el label basado en la predicción
    label = labels.get(prediction, "Desconocido")

    # Generar el heatmap
    heatmap = grad_cam(array)

    return (label, proba, heatmap)

# Función para leer archivos DICOM
def read_dicom_file(path):
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB

# Función para leer archivos JPG o PNG
def read_image_file(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img_array = np.array(img)
    return img_array

# Función para generar reporte PDF
def generate_pdf(patient_id, label, proba, original_image, heatmap_image):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Reporte diagnóstico de lesiones óseas", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"ID del paciente: {patient_id}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Zona afectada: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Probabilidad de lesión crítica: {proba:.2f}%", ln=True)

    pdf.ln(10)

    # Convertir imágenes a PIL para agregar al PDF
    original_image_pil = Image.fromarray(original_image)
    heatmap_image_pil = Image.fromarray(heatmap_image)

    # Guardar imágenes en buffers de memoria
    original_image_buffer = BytesIO()
    heatmap_image_buffer = BytesIO()

    original_image_pil.save(original_image_buffer, format="PNG")
    heatmap_image_pil.save(heatmap_image_buffer, format="PNG")

    # Volver a la posición inicial del buffer
    original_image_buffer.seek(0)
    heatmap_image_buffer.seek(0)

    # Convertir los buffers de imágenes a archivos temporales
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as original_temp_file:
        original_temp_file.write(original_image_buffer.getvalue())
        original_temp_file_path = original_temp_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as heatmap_temp_file:
        heatmap_temp_file.write(heatmap_image_buffer.getvalue())
        heatmap_temp_file_path = heatmap_temp_file.name

    # Agregar las imágenes desde archivos temporales
    pdf.image(original_temp_file_path, x=10, y=80, w=90)
    pdf.image(heatmap_temp_file_path, x=110, y=80, w=90)

    pdf.ln(85)
    pdf.cell(200, 10, txt="Imagen Original", ln=False, align="C")
    pdf.cell(200, 10, txt="Heatmap de Imagen", ln=False, align="C")

    # Guardar el PDF en un buffer de memoria y devolverlo
    pdf_buffer = BytesIO()
    pdf_buffer.write(pdf.output(dest='S').encode("latin1"))

    # Regresar los bytes del PDF para descargar
    pdf_buffer.seek(0)
    
    # Eliminar archivos temporales
    os.remove(original_temp_file_path)
    os.remove(heatmap_temp_file_path)
    
    return pdf_buffer

# Interfaz en Streamlit
def main():
    # Inicializar el estado de la sesión
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
        st.session_state.label = None
        st.session_state.proba = None
        st.session_state.heatmap = None
        st.session_state.pdf_buffer = None  # Estado para el PDF generado
    
    st.title("🚀🩺Herramienta para diagnóstico rápido de lesiones óseas🦴🧠")

    # Entrada para la identificación del paciente

    patient_id = st.text_input("ID del paciente:")

    # Cargar array
    uploaded_file = st.file_uploader("📁 Cargar Imágen (DICOM, JPG, PNG)", type=["dcm", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".dcm":
            st.session_state.image_array = read_dicom_file(uploaded_file)
        else:
            st.session_state.image_array = read_image_file(uploaded_file)
        
        # Mostrar array original
        st.image(st.session_state.image_array, caption="🖼 Imágen Radiográfica cargada", use_column_width=True)

        if st.button("🤖 Predecir"):
            st.session_state.label, st.session_state.proba, st.session_state.heatmap = predict(st.session_state.image_array)
            
            # Mostrar resultados
            st.write(f"Zona afectada: {st.session_state.label}")
            st.write(f"Probabilidad de lesion crítica: {st.session_state.proba:.2f}%")
            
            # Mostrar heatmap
            st.image(st.session_state.heatmap, caption="🔥 Imágen Radiográfica con zonas afectadas", use_column_width=True)

            # Generar el PDF solo si se ha realizado una predicción
            if st.session_state.label is not None:
                st.session_state.pdf_buffer = generate_pdf(
                    patient_id,
                    st.session_state.label,
                    st.session_state.proba,
                    st.session_state.image_array,
                    st.session_state.heatmap
                )
        
    # Botón de descarga del PDF (fuera del flujo de predicción)
    if st.session_state.pdf_buffer is not None:
        st.download_button(
            label="📄 Descargar Reporte en PDF",
            data=st.session_state.pdf_buffer,
            file_name=f"Diagnóstico{patient_id}.pdf",
            mime="application/pdf"
        )

    # Botón para reiniciar la aplicación
    if st.button("🔄 Reiniciar Aplicación"):
        st.rerun()

if __name__ == "__main__":
    main()
