## 🚀🩺Herramienta de apoyo para dianóstico rápido de lesiones oseas🦴🧠
### Por: Santiago García Solarte & Santiago Loaiza Cardona
### Entrega proyecto final: Desarrollo de proyectos de inteligencia artificial.

Deep Learning aplicado en el procesamiento de imágenes radiográficas de extremidades, utilizando el dataset MURA (Musculoskeletal Radiographs) de Stanford (consultalo [AQUI](https://stanfordmlgroup.github.io/competitions/mura/)), con el fin de clasificarlas con una probabilidad de lesion critica en las siguientes categorias:

- Codo
- Dedo
- Antebrazo
- Mano
- Humero
- Hombro
- Muñeca

Adicional se realiza la aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta:
Realiza los siguientes pasos para empezar a utilizarla:

#### Metodo #1: Visual Studio Code

Requerimientos necesarios para el funcionamiento:

- Instale Visual Studio Code para windows [AQUI](https://code.visualstudio.com/download) 
  
- Abra un entorno de ejecucion de "powershell"

- Ejecute los siguientes comandos:

      git clone https://github.com/santenana/Proyecto_final

      cd Proyecto_final

      pip install -r requirements.txt

    Verificar que las dependencias requeridas se hayan instalado con exito.

## Pruebas en contenedor Docker

Para ejecutar la herramienta de diagnostico, ejecute los siguientes comandos:

        cd Proyecto_final

        docker build -t streamlit-app .

Iniciará el proceso de crear la imagen con la información requerida. Finalizado el proceso de creacion ejecuta:

    docker run -p 8501:8501 streamlit-app     # "streamlit-app" sera el nombre de la imagen creada.

Ahora abre el navegador web de tu preferencia e ingresa la siguiente URL:

        http://localhost:8501/

En este punto se estará ejecutando el aplicativo de diagnostivo medico para lesiones osea. A continuacion veras el funcionamiento de la interfaz gráfica, la cual es muy intuitiva.

#### Uso de la Interfaz Gráfica:

- Ingrese el documento que identifica al paciente en la caja de texto.
- Presione el botón 'Cargar imagen', seleccione la imagen desde el explorador de archivos de su computador.
- Desde el repositorio podra descargar imagenes de prueba.
- Se mostrará la imagen que se ha seleccionado.
- Presione el botón 'Descargar Reporte en PDF' y automaticamente se descargara un documento PDF con el diagnóstico del modelo.
- Presión el botón 'Reiniciar Aplicación' para realizar un nuevo diagnóstico.

Para ver el funcionamiento del aplicativo has clic [AQUI](Video_muestra/streamlit.mp4)

---

## Arquitectura de archivos propuesta.
## streamlit_app.py

Contiene el diseño de la interfaz gráfica utilizando Streamlit. Los botones llaman métodos contenidos dentro del script.

## streamlit.py

Es la interfaz que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM, ademas de ejecutar las funciones de fondo para generar el reporte PDF.

## read_img.py

Script que lee la imagen en formato DICOM, PNG, JPG o JPEG para visualizarla en la interfaz gráfica.

## preprocess_img.py

Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 1285x128
- conversión de BGR a RGB para obtener los 3 canales de color.
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor [1,128,128,3])

## load_model.py

Código que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado 'clasificador_fractura.h5'.

## grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es la siguiente:

        model = Sequential()

        model.add(Conv2D(32,(3,3),1, activation="relu", input_shape=(128,128,3)))
        model.add(MaxPooling2D())

        model.add(Conv2D(64,(3,3),1,activation="relu"))
        model.add(MaxPooling2D())

        model.add(Conv2D(32,(3,3),1,activation="relu"))
        model.add(MaxPooling2D())

        model.add(Flatten())    
        model.add(Dense(128, activation='relu'))
        model.add(Dense(7,activation='softmax'))

 La funcion grad_cam esta asociada a la capa "conv2d_2" ya que es la capa que mejor resultado se obtuvo. La arquitectura del modelo incluye múltiples bloques convolucionales, seguidos de capas de max pooling y dense con un entrenamiento de 20 ciclos.


## Acerca de Grad-CAM

Grad-CAM es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

## Pruebas Unitarias

Para ejecutar las pruebas unitarias, asegúrate de tener las dependencias instaladas, ejecuta el siguiente comando:


---

## Desarrollo del Proyecto:
Santiago García Solarte - https://github.com/santenana

Santiago Loaiza Cardona- https://github.com/S-loaiza-UAO
