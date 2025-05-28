# ProyectoWeb

Este proyecto tiene como objetivo crear un **predictor de potabilidad de agua superficial** en la nube. Para ello, trabajaremos con **PythonAnywhere** como entorno de despliegue.

## 📁 Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas y archivos:

data/
│── Contiene los archivos .csv con la información de agua.

modelos/
│── Incluye el archivo model_reg.pkl.

requirements.txt
│── Lista de requerimientos necesarios para trabajar con el predictor.

api_marketing.py
│── Código de la API para interactuar con el modelo.

pythonanywhere_com_wsgi.py
│── Archivo de configuración para desplegar en PythonAnywhere.

index.html
│── Código HTML para la interfaz web.

## ⚙️ Instalación y Uso

### Requisitos Previos

- Tener Python 3.13 instalado.
- Tener una cuenta en [PythonAnywhere](https://www.pythonanywhere.com/).
- pip (gestor de paquetes de Python).

### Instalación Local

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu_usuario/ProyectoWeb.git
   cd ProyectoWeb
   
2. Instala los requisitos
    ```bash
    pip install -r requirements.txt

3. Ejecuta la API localmente:
    ```bash
    python api_marketing.py

4. **Abre `index.html`** en tu navegador para usar la interfaz web.

---

### 🚀 Despliegue en PythonAnywhere

1. **Sube los archivos del proyecto** a tu cuenta de [PythonAnywhere](https://www.pythonanywhere.com/).

2. **Configura el archivo** `pythonanywhere_com_wsgi.py` en tu panel de Web.

3. **Instala los paquetes necesarios** desde `requirements.txt` usando la consola de PythonAnywhere:

   ```bash
   pip install -r requirements.txt



