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

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Ruta del modelo
model_path = "deteccion_fracturas.h5"

# Cargar el modelo
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Modelo cargado exitosamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Función para preprocesar la imagen
def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array

# Función Grad-CAM
def grad_cam(array):
    img = preprocess(array)
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer("conv10_thisone")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)
    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    return superimposed_img[:, :, ::-1]

# Función de predicción
def predict(array):
    batch_array_img = preprocess(array)
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    elif prediction == 1:
        label = "normal"
    elif prediction == 2:
        label = "viral"
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

# Función para generar el reporte en PDF
def generate_pdf(patient_id, label, proba, original_image, heatmap_image):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Reporte Diagnóstico Médico", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Cédula del Paciente: {patient_id}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Resultado de la Predicción: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Probabilidad: {proba:.2f}%", ln=True)

    pdf.ln(10)
    # Guardar imágenes en archivos temporales
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        original_image_path = temp_file.name
        Image.fromarray(original_image).save(original_image_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        heatmap_image_path = temp_file.name
        Image.fromarray(heatmap_image).save(heatmap_image_path)

    # Agregar las imágenes al PDF
    pdf.image(original_image_path, x=10, y=80, w=90)
    pdf.image(heatmap_image_path, x=110, y=80, w=90)

    pdf.ln(85)
    pdf.cell(200, 10, txt="Imagen Original", ln=False, align="C")
    pdf.cell(200, 10, txt="Heatmap de Grad-CAM", ln=False, align="C")

    # Definir ruta de almacenamiento
    save_directory = "reportes_pdf"  # Directorio para guardar los PDFs
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Guardar el PDF con el ID del paciente como nombre
    pdf_filename = f"{save_directory}/reporte_{patient_id}.pdf"
    pdf.output(pdf_filename)

    return pdf_filename

# Interfaz en Streamlit
def main():
    st.title("🚀🩺Herramienta de apoyo para dianóstico rápido de lesiones oseas🦴🧠")

    # Entrada para la identificación del paciente
    patient_id = st.text_input("Ingrese el ID del paciente:")

    # Cargar imagen
    uploaded_file = st.file_uploader("Cargar imagen (DICOM, JPG, PNG)", type=["dcm", "jpg", "jpeg", "png"])

    # Verificar si ya se ha cargado una imagen previamente y limpiar si es necesario
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
        st.session_state.label = None
        st.session_state.proba = None
        st.session_state.heatmap = None

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".dcm":
            st.session_state.image_array = read_dicom_file(uploaded_file)
        else:
            st.session_state.image_array = read_image_file(uploaded_file)
        
        # Mostrar imagen original
        st.image(st.session_state.image_array, caption="Imagen Radiográfica cargada", use_column_width=True)

        if st.button("Predecir..."):
            st.session_state.label, st.session_state.proba, st.session_state.heatmap = predict(st.session_state.image_array)
            
            # Mostrar resultados
            st.write(f"Resultado: {st.session_state.label}")
            st.write(f"Probabilidad: {st.session_state.proba:.2f}%")
            
            # Mostrar heatmap
            st.image(st.session_state.heatmap, caption="Imagen Radiográficas con zonas afectadas", use_column_width=True)

            # Botón para descargar el reporte en PDF
            if st.button("Generar Reporte PDF..."):
                pdf_path = generate_pdf(patient_id, st.session_state.label, st.session_state.proba, st.session_state.image_array, st.session_state.heatmap)
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Descargar Reporte en PDF",
                        data=file,
                        file_name=f"reporte_{patient_id}.pdf",
                        mime="application/pdf"
                    )

        # Botón para eliminar la información cargada
        if st.button("Eliminar Información"):
            st.session_state.image_array = None
            st.session_state.label = None
            st.session_state.proba = None
            st.session_state.heatmap = None
            st.experimental_rerun()  # Recarga la aplicación para limpiar los datos

    else:
        st.write("Por favor, cargue una imagen para realizar el diagnóstico.")

if __name__ == "__main__":
    main()
