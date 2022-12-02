# Infrastructure

In this folder you can add infrastructure documents/guides including:

* Docker/kubernetes setup and management.

Teniendo en cuenta lo aclarado en clase, esta parte no se diligencia por ahora debido a que este tema no se ha visto.

* Server-based configuration (minimal system resources, VMs setup, webserver setup, among others).

Teniendo en cuenta que el modelo se ha construido y entrenado en GoogleColab, los siguientes son los recursos asignados:
# Distribución de Linux:

![WhatsApp Image 2022-12-01 at 9 33 37 PM](https://user-images.githubusercontent.com/73256719/205202540-7f00e3e9-1c2e-4d32-b8e6-92479f7b5d9d.jpeg)

# GPU asignada:

![WhatsApp Image 2022-12-01 at 9 08 40 PM](https://user-images.githubusercontent.com/73256719/205202727-29d1d222-4a83-4d8b-a5db-bf8f2a1fd6c4.jpeg)


# Disco Asignado:

![WhatsApp Image 2022-12-01 at 9 33 36 PM](https://user-images.githubusercontent.com/73256719/205202608-33c2f1ad-0c1b-4aef-a24d-0ac68de06090.jpeg)


* Environment configuration (pyenv, poetry, jupyter, rstudio).

Para la estructuración e instalación del proyecto se utilizará poetry, destacando la siguiente secuencia de comandos principales:

poetry ini. Para la inicialización del proyecto.

poetry add. Para adicionar paquetes, los cuales son incluidos como dependencias de forma autmática en el archivo pyproject.toml.

poetry update. Para actualizar el archivo poetry.lock con las ultimas versiones de dependencias instaladas. 

poetry build. Para construir archivo .whl con el que se publicará el proyecto.

poetry install. Para resolver dependencias incluidas en achivo pyproject. toml e instalar el proyecto.



* Execution pipelines (airflow, mlflow).

Para el despliegue se utilizará la plataforma MLFlow.
