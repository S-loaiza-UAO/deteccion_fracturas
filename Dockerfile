# Usamos una imagen base de Python específica
FROM python:3.10

# Instalamos las dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-opencv \
    gnome-screenshot \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo en el contenedor
WORKDIR /home/src

# Copia los archivos necesarios al contenedor
COPY requirements.txt .

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos al contenedor
COPY . .

# Exponer el puerto
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]