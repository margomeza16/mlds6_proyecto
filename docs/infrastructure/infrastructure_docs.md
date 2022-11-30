# Infrastructure

In this folder you can add infrastructure documents/guides including:

* Docker/kubernetes setup and management.
* Server-based configuration (minimal system resources, VMs setup, webserver setup, among others).

Teniendo en cuenta que el modelo se ha construido y entrenado en GoogleColab, los siguientes son los recursos asignados:
# Distribución de Linux:
NAME="Ubuntu"

VERSION="18.04.6 LTS (Bionic Beaver)"

ID=ubuntu

ID_LIKE=debian

PRETTY_NAME="Ubuntu 18.04.6 LTS"

VERSION_ID="18.04"

HOME_URL="https://www.ubuntu.com/"

SUPPORT_URL="https://help.ubuntu.com/"

BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"

PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"

VERSION_CODENAME=bionic

UBUNTU_CODENAME=bionic

# GPU asignada:

+-----------------------------------------------------------------------------+

| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |

|-------------------------------+----------------------+----------------------+

| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |

| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |

|                               |                      |               MIG M. |

|===============================+======================+======================|

|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |

| N/A   38C    P8     8W /  70W |      0MiB / 15109MiB |      0%      Default |

|                               |                      |                  N/A |

+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+

| Processes:                                                                  |

|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |

|        ID   ID                                                   Usage      |

|=============================================================================|

|  No running processes found                                                 |

+-----------------------------------------------------------------------------+

# Disco Asignado:

Filesystem      Size  Used Avail Use% Mounted on

overlay          79G   23G   56G  29% /

tmpfs            64M     0   64M   0% /dev

shm             5.7G     0  5.7G   0% /dev/shm

/dev/root       2.0G  1.1G  910M  54% /sbin/docker-init

tmpfs           6.4G   48K  6.4G   1% /var/colab

/dev/sda1        75G   41G   35G  55% /opt/bin/.nvidia

tmpfs           6.4G     0  6.4G   0% /proc/acpi

tmpfs           6.4G     0  6.4G   0% /proc/scsi

tmpfs           6.4G     0  6.4G   0% /sys/firmware


* Environment configuration (pyenv, poetry, jupyter, rstudio).
* Execution pipelines (airflow, mlflow).

Para la estructuración e instalación del proyecto se utilizará poetry, destacando la siguiente secuencia de comandos principales:

poetry ini. Para la inicialización del proyecto.

poetry add. Para adicionar paquetes, los cuales son incluidos como dependencias de forma autmática en el archivo pyproject.toml.

poetry update. Para ctualizar el archivo poetry.lock con las ultimas versiones de dependencias instaladas. 

poetry build. Para construir archivo .whl con el que se publicará el proyecto.
petry install. Para resolver dependencias incluidas en achivo pyproject. toml e instalar el proyecto.
