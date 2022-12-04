# Infrastructure

In this folder you can add infrastructure documents/guides including:

* Docker/kubernetes setup and management.

Teniendo en cuenta lo aclarado en clase, esta parte no se diligencia por ahora debido a que este tema no se ha visto.

* Server-based configuration (minimal system resources, VMs setup, webserver setup, among others).

# Configuración Virtual Machine (VM) Linux

![VM](https://user-images.githubusercontent.com/73256719/205458569-b65cd603-1f30-4e46-a0f9-35aa9f48d455.png)

Teniendo en cuenta que el modelo se ha construido y entrenado en GoogleColab, los siguientes son los recursos asignados:
# Distribución de Linux:

![WhatsApp Image 2022-12-01 at 9 33 36 PM](https://user-images.githubusercontent.com/73256719/205207436-89ec0594-c1e2-4da2-a61f-5e5793557032.jpeg)

# GPU asignada:

![WhatsApp Image 2022-12-01 at 9 08 41 PM (2)](https://user-images.githubusercontent.com/73256719/205207648-6510d3dd-5ebf-46f0-9e8b-d18cfa01565e.jpeg)

# Disco Asignado:

![WhatsApp Image 2022-12-01 at 9 51 48 PM](https://user-images.githubusercontent.com/73256719/205207528-70361893-2b8c-4235-8604-da939058dc9a.jpeg)

* Environment configuration (pyenv, poetry, jupyter, rstudio).

Para la estructuración e instalación del proyecto se utilizará poetry, destacando la siguiente secuencia de comandos principales:

poetry ini. Para la inicialización del proyecto. Inicializa archivo pyproject.toml

poetry add. Para adicionar paquetes, los cuales son incluidos como dependencias de forma autmática en el archivo pyproject.toml.

poetry update. Para actualizar el archivo poetry.lock con las ultimas versiones de dependencias instaladas. 

poetry build. Para construir archivo .whl con el que se publicará el proyecto.

poetry install. Para resolver dependencias incluidas en achivo pyproject. toml e instalar el proyecto.



* Execution pipelines (airflow, mlflow).

Para el despliegue y ejecución del modelo se utilizará la libreria MLFlow, mediante los siguientes comandos básicos:

  Inicialización del proyecto mediante la configuración del archivo **MLproject**

  Creación de experimientos mediante **mlflow experiments**

  Ejecución del proyecto mediante **mlflow run**

  Generación del dashboard mediante **mlflow server**

  Despliegue del modelo mediante **mlflow models serve**


