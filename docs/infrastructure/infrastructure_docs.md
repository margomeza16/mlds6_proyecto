# Infrastructure

In this folder you can add infrastructure documents/guides including:

* Docker/kubernetes setup and management.

Teniendo en cuenta lo aclarado en clase, esta parte no se diligencia por ahora debido a que este tema no se ha visto.

* Server-based configuration (minimal system resources, VMs setup, webserver setup, among others).

# Configuración Virtual Machine (VM) Linux

OS: Debian GNU/Linux 11 (bullseye) x86_64
Host: Google Compute Engine
Kernel: 5.10.0-19-cloud-amd64
Uptime: 29 days, 20 hours, 29 mins
Packages: 769 (dpkg)
Shell: bash 5.1.4
Terminal: /dev/pts/0
CPU: Intel Xeon (2) @ 2.199GHz
Memory: 131MiB / 3930MiB

Teniendo en cuenta que el modelo se ha construido y entrenado en GoogleColab, los siguientes son los recursos asignados:
# Distribución de Linux:

![WhatsApp Image 2022-12-01 at 9 33 36 PM](https://user-images.githubusercontent.com/73256719/205207436-89ec0594-c1e2-4da2-a61f-5e5793557032.jpeg)

# GPU asignada:

![WhatsApp Image 2022-12-01 at 9 08 41 PM (2)](https://user-images.githubusercontent.com/73256719/205207648-6510d3dd-5ebf-46f0-9e8b-d18cfa01565e.jpeg)

# Disco Asignado:

![WhatsApp Image 2022-12-01 at 9 51 48 PM](https://user-images.githubusercontent.com/73256719/205207528-70361893-2b8c-4235-8604-da939058dc9a.jpeg)

* Environment configuration (pyenv, poetry, jupyter, rstudio).

Para la estructuración e instalación del proyecto se utilizará poetry, destacando la siguiente secuencia de comandos principales:

poetry ini. Para la inicialización del proyecto.

poetry add. Para adicionar paquetes, los cuales son incluidos como dependencias de forma autmática en el archivo pyproject.toml.

poetry update. Para actualizar el archivo poetry.lock con las ultimas versiones de dependencias instaladas. 

poetry build. Para construir archivo .whl con el que se publicará el proyecto.

poetry install. Para resolver dependencias incluidas en achivo pyproject. toml e instalar el proyecto.



* Execution pipelines (airflow, mlflow).

Para el despliegue se utilizará la libreria MLFlow.
