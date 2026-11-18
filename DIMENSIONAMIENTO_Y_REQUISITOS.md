# Dimensionamiento y requisitos de software – BaNCS Vision / DayTransactionsPredictor

> Estado: **propuesta preliminar**. Los cálculos de SQL Server usan el esquema preliminar compartido; los supuestos marcados con ⚠️ deben confirmarse con datos reales.
> Las versiones de software son de referencia: antes de desplegar hay que **fijarlas con `pip freeze`** del entorno donde hoy funciona el modelo, porque no pude leer las versiones instaladas.

## 1. Arquitectura que se dimensiona (según el diagrama del banco)

- **Fuentes:** FTP BaNCS (reportes IMG), SharePoint BP (reportes Reposting) y reportes Control M.
- **AKS (Linux, nube del banco, "Server BaNCS Vision"):** Python, Spark, Airflow, MLflow y Docker. Aquí viven los "Resumen" de cada fuente, el **Motor BaNCS Vision**, el dashboard (vista estática y vista conversacional) y los guardrails.
- **SQL Server 2025 (on-premise):** solo **tablas de consumo**. El motor escribe ahí y el dashboard lee de ahí.
- **LLM gpt-4.1 (tenant BP):** recibe el esquema y los metadatos y devuelve código. Los datos no se envían al modelo.
- **Usuarios y roles:** Entra ID.
- Todo dentro de la Red BP.

> Este diagrama difiere de la primera propuesta en [ARQUITECTURA.md](./ARQUITECTURA.md): Spark, Airflow y el procesamiento corren en AKS y no en un servidor local, y el almacén final es SQL Server. Conviene actualizar ese documento.

## 2. Dimensionamiento de SQL Server 2025

### 2.1 Método

```
bytes por fila = datos de la fila + overhead de fila + entrada de slot (2 B) + filas de cada índice
datos de la fila = columnas fijas + columnas variables (promedio real)
overhead de fila = 4 (cabecera) + 2 + ceil(columnas/8) (bitmap de nulos) + 2 + 2 por columna variable
fila de índice   = bytes de la clave + bytes de la clave del clustered (4) + 9 de overhead
tamaño/año = filas por día × 365 × bytes por fila
```

Tamaños de tipos en SQL Server: `INT` 4, `DATETIME` 8 (`DATETIME2(0)` 6), `DATE` 3, `DECIMAL(8,2)` 5, `BIT` 1 (hasta 8 bits comparten un byte), `VARCHAR(n)` longitud real + 2.

### 2.2 Cálculo por tabla

| Tabla | Filas por día | Bytes por fila (dato + índices) | Detalle | Crecimiento por año |
|---|---|---|---|---|
| `transacciones_historicas` | 48 (24 h × 2 tipos) | ≈ 138 | fila 52 + UNIQUE 36 + `idx_tipo` 28 + `idx_fecha` 22 | **2.4 MB** |
| `transacciones_por_cola_historicas` | 480 (24 h × 20 colas) | ≈ 137 | fila 71 + UNIQUE 26 + `idx_fecha` 22 + `idx_cola` 18 | **24 MB** |
| `colas` | 0 | ≈ 500 | 20 filas fijas | < 0.1 MB (fijo) |
| `procesos_criticos` ⚠️ | 50 | ≈ 56 | fila 40 + `idx_fecha` 16 | **1.0 MB** |
| `reposting` ⚠️ | 100 | ≈ 73 | fila 57 + `idx_fecha` 16 | **2.7 MB** |
| `calendario_laboral` | 1 | ≈ 62 | fila 46 + UNIQUE 16 | **0.02 MB** |
| `backtest_resultados` ⚠️ | 100 | ≈ 70 | sin índices | **2.6 MB** |
| **Subtotal datos históricos** | | | | **≈ 33 MB por año** |

Tablas de pronóstico. Cada corrida completa (365 días hacia adelante) ocupa:

| Tabla | Filas por corrida | Bytes por fila | Tamaño por corrida |
|---|---|---|---|
| `pronosticos_tph` | 8,760 | ≈ 104 | **0.9 MB** |
| `pronosticos_colas` | 175,200 (8,760 × 20) | ≈ 111 | **19.4 MB** |

Hay dos escenarios para los pronósticos:

- **A. Solo la última corrida** (el esquema actual, con `UNIQUE(fecha_hora)`, sobrescribe): ≈ **20 MB** fijos.
- **B. Guardar una corrida diaria para comparar pronóstico contra real** (requiere agregar `version_modelo` o `fecha_corrida` a la clave única): 20.3 MB × 365 = **≈ 7.4 GB por año**.

### 2.3 Resultado

Base actual (≈ 1.7 años ya cargados) ≈ **50 MB**.

| Horizonte | Escenario A (solo última corrida) | Escenario B (corrida diaria) |
|---|---|---|
| Hoy | ≈ 50 MB | ≈ 50 MB |
| +1 año (mínimo pedido) | ≈ 0.1 GB | ≈ 7.6 GB |
| +3 años | ≈ 0.2 GB | ≈ 22 GB |
| +5 años | ≈ 0.25 GB | ≈ 37 GB |

**Conclusión:** los datos son muy pequeños. El tamaño lo define si se conserva historial de pronósticos, no las tablas históricas.

### 2.4 Aprovisionamiento recomendado (con holgura para el escenario B a 3 años)

| Recurso | Valor | Motivo |
|---|---|---|
| Edición | Standard | La carga es mínima; verificar los límites de la edición Express antes de considerarla. |
| CPU / RAM | 4 vCPU / 16 GB (`max server memory` ≈ 12 GB) | Consultas generadas por el LLM sobre tablas pequeñas; 8 vCPU / 32 GB solo si comparte la instancia con otras bases. |
| Archivo de datos | 50 GB | Escenario B a 3 años (≈ 22 GB) con holgura de 2× y autogrowth de 512 MB. |
| Log de transacciones | 10 GB | Cargas por lotes diarias; el modelo de recuperación FULL requiere backups de log. |
| `tempdb` | 10 GB | Ordenamientos y agregaciones del dashboard. |
| Backups | 100 GB | Full semanal + diferencial diario + log, retención 30 días. |
| Disco total | ≈ 170 GB | Separar volúmenes de datos, log y backups. |

Mínimo viable si solo se guarda la última corrida: 10 GB de datos, 5 GB de log y 20 GB de backups.

### 2.5 Observaciones sobre el esquema preliminar

1. **Sintaxis de MySQL:** `AUTO_INCREMENT`, `INDEX` dentro del `CREATE TABLE`, `BOOLEAN`, `ON UPDATE CURRENT_TIMESTAMP` y `TEXT` no funcionan en SQL Server. Hay que usar `IDENTITY`, `CREATE INDEX`, `BIT`, un trigger o columna manual para actualización, y `VARCHAR(MAX)` o `NVARCHAR`. `DEFAULT CURRENT_TIMESTAMP` sí funciona; se recomienda `SYSDATETIME()`.
2. **`transacciones_historicas`:** `UNIQUE(fecha_hora)` impide guardar financiero y no financiero en la misma hora. Debe ser `UNIQUE(fecha_hora, tipo_transaccion)`.
3. **`procesos_criticos`:** `UNIQUE(fecha)` impide varios procesos el mismo día; debe ser `UNIQUE(fecha, proceso_id)`.
4. **Índices redundantes:** `idx_fecha` duplica al `UNIQUE` que empieza por `fecha_hora`, y `idx_tipo` tiene muy poca selectividad. Eliminarlos baja el tamaño y acelera las cargas.
5. **`cola_nombre` en `transacciones_por_cola_historicas`** se repite en cada fila y ya existe en `colas`. Quitarlo ahorra ≈ 30 B por fila (≈ 20 % de la tabla más grande).
6. **Rutas de entrada:** el diagrama muestra que las fuentes son FTP, SharePoint y Control M, no los Excel de `data/raw`. Habrá que ajustar las tablas de origen y sus columnas.
7. **Para el LLM:** crear un esquema aparte (por ejemplo `consumo`) con vistas, un usuario de **solo lectura** para el dashboard y un usuario distinto de escritura para el motor. Así los guardrails se refuerzan en la base de datos.
8. **Compresión:** `PAGE` compression reduce las tablas de colas entre 50 % y 70 %; es opcional dado el tamaño.

## 3. Dimensionamiento de AKS

### 3.1 Cargas y requests estimados

| Carga | Réplicas | vCPU | RAM |
|---|---|---|---|
| Airflow (scheduler + webserver + triggerer) | 1 | 1.75 | 4 GB |
| Base de metadatos de Airflow y MLflow (PostgreSQL) | 1 | 0.5 | 1 GB |
| MLflow Tracking Server | 1 | 0.5 | 1 GB |
| Dashboard web (vista estática) | 2 | 1 | 2 GB |
| API conversacional + guardrails | 2 | 2 | 4 GB |
| Sandbox de ejecución del código generado | 1 | 1 | 2 GB |
| Motor BaNCS Vision / Resúmenes (pods por tarea) | pico 2 | 2 | 4 GB |
| Spark (driver + 2 ejecutores) | pico | 6 | 16 GB |
| Ingress, monitoreo y agentes | 1 | 1 | 2 GB |
| **Total en pico** | | **≈ 16 vCPU** | **≈ 36 GB** |

Nota: con volúmenes de decenas de MB, Spark corre bien en **modo local** dentro de un solo pod (4 vCPU / 8 GB). El clúster de Spark solo se justifica si el banco pide estandarizar o si los reportes IMG crecen mucho.

### 3.2 Node pools (producción)

| Pool | VM (referencia Azure) | Nodos | Uso |
|---|---|---|---|
| `system` | `Standard_D2s_v5` (2 vCPU / 8 GB) | 2 (3 si se quiere distribución en zonas) | Componentes de AKS, CoreDNS, ingress |
| `apps` | `Standard_D4s_v5` (4 vCPU / 16 GB) | 2 | Airflow, MLflow, dashboard, API, sandbox |
| `jobs` | `Standard_D8s_v5` (8 vCPU / 32 GB) | autoscaling 1–3 | Spark, entrenamiento (LightGBM, statsmodels), resúmenes |

Capacidad en pico: ≈ 28 vCPU / 112 GB, con margen sobre los 16 vCPU / 36 GB estimados y los ≈ 10 % reservados por nodo para el sistema. Dev/QA: un solo nodo `D4s_v5` por ambiente.

### 3.3 Almacenamiento

| Elemento | Tamaño | Tipo |
|---|---|---|
| PostgreSQL (metadatos) | 32 GiB | Premium SSD (PVC) |
| Logs de Airflow | 50 GiB | PVC, o logging remoto a Blob |
| Artefactos de MLflow (modelos y gráficos) | 50 GiB | Azure Blob o PVC |
| Zona de aterrizaje de reportes (IMG, Reposting, Control M) ⚠️ | 100 GiB (≈ 100 MB/día × 365) | PVC o Azure Files |
| Scratch de Spark | 128 GB de disco del nodo | Efímero |
| Azure Container Registry | Standard (100 GB) | Imágenes de 1.5–3 GB cada una |

### 3.4 Red

- Ingress interno (balanceador interno) con DNS privado y certificado de la CA del banco.
- Conectividad AKS → SQL Server on-premise por ExpressRoute o VPN, puerto 1433, abierta solo desde la subred de AKS.
- AKS → gpt-4.1 en el tenant BP mediante endpoint privado.
- AKS → FTP BaNCS y SharePoint BP: abrir los puertos requeridos.
- NetworkPolicies entre namespaces y **egress restringido** para el sandbox.

### 3.5 Identidad y seguridad

- Entra ID por OIDC en el dashboard y grupos por rol (el rol se valida en la API).
- Workload Identity para que los pods accedan a Key Vault y Blob sin contraseñas.
- Secretos en Azure Key Vault (con el driver CSI) o el vault del banco.
- Cuenta de SQL de solo lectura para el dashboard y otra de escritura para el motor.

## 4. Software necesario

### 4.1 Plataforma

| Componente | Versión de referencia | Para qué |
|---|---|---|
| Azure Kubernetes Service | Versión soportada vigente (1.30 o superior; verificar la disponible en el banco) | Orquestación |
| Nodos | Linux (Ubuntu / Azure Linux) | |
| kubectl, Helm | 1.30+ / 3.14+ | Administración y despliegue |
| Docker / BuildKit | 24+ | Construcción de imágenes |
| Azure Container Registry | Standard | Registro de imágenes |
| SQL Server | **2025** (compatibilidad nivel 170) | Tablas de consumo |
| ODBC Driver for SQL Server | **18** (`msodbcsql18`) | Conexión desde Python |
| Git, CI/CD (Azure DevOps o equivalente) | | Versionado y despliegue |

### 4.2 Runtime

| Componente | Versión de referencia | Para qué |
|---|---|---|
| **Python** | **3.11** (3.12 también compatible con las librerías abajo) | Modelo, API, Airflow |
| **Java** | OpenJDK **17** | Requerido por Spark |
| **Apache Spark / PySpark** | **3.5.x** (línea estable; Spark 4.x requiere validar compatibilidad) | Procesamiento |
| **Apache Airflow** | **2.10.x** (soporta Python 3.8–3.12) | Orquestación |
| **MLflow** | **2.x** (última estable) | Registro de experimentos y modelos |
| Spark Operator (Kubeflow) | Última estable | Ejecutar Spark en Kubernetes (solo si no se usa modo local) |
| PostgreSQL | 15 o 16 | Metadatos de Airflow y MLflow (Airflow no usa SQL Server como metadata) |
| Nginx Ingress Controller | Última estable | Entrada HTTPS |

### 4.3 Librerías de Python

Ya en [requirements.txt](./requirements.txt) (sin versiones fijadas):

| Librería | Rango de referencia | Para qué |
|---|---|---|
| pandas | 2.2.x | Manejo de datos |
| numpy | 1.26.x – 2.x | Cálculo numérico |
| scipy | 1.13+ | Estadística |
| scikit-learn | 1.5+ | Métricas y modelos |
| lightgbm | 4.x | Modelo de pronóstico |
| statsmodels | 0.14.x | Modelos estadísticos |
| holidays | 0.5x+ | Calendario de festivos |
| openpyxl | 3.1.x | Leer Excel |
| matplotlib, pillow | 3.9 / 10–11 | Gráficos e imágenes |
| jupyter | — | **Solo desarrollo; excluir de la imagen del servidor** |

Nuevas, requeridas por la arquitectura:

| Librería | Para qué |
|---|---|
| pyodbc 5.x + SQLAlchemy 2.x | Leer y escribir en SQL Server |
| pyspark 3.5.x | Spark desde Python |
| apache-airflow 2.10.x y providers (`microsoft-mssql`, `cncf-kubernetes`, `apache-spark`) | DAGs |
| mlflow | Seguimiento de modelos |
| fastapi + uvicorn + pydantic 2.x | API del dashboard y de la vista conversacional |
| openai (SDK 1.x) | Llamar a gpt-4.1 |
| azure-identity, msal | Entra ID y Workload Identity |
| azure-keyvault-secrets, azure-storage-blob | Secretos y archivos |
| Parser de reportes (a definir según el formato real de IMG/Control M: `pdfplumber`, `pytesseract` u otro) ⚠️ | Extraer los "Resumen" |
| Librería de guardrails (por ejemplo `sqlglot` para validar SQL, o un framework de guardrails) | Validación del código y las consultas generadas |
| pytest, ruff, mypy | Pruebas y calidad |

### 4.4 Imágenes base sugeridas

- `python:3.11-slim-bookworm` + `msodbcsql18` + OpenJDK 17 para Motor, API y Airflow workers.
- Imagen oficial de Spark 3.5.x para ejecutores en Kubernetes.
- Dependencias **pineadas** con `pip-compile` o `uv pip compile` y un archivo de constraints, más el de Airflow para sus dependencias.

## 5. Decisiones y datos que faltan para cerrar el dimensionamiento

1. Cantidad real de procesos críticos por día, tipos de reposting y filas del backtest (⚠️ en las tablas).
2. Tamaño y formato de los reportes de FTP, SharePoint y Control M, y frecuencia de llegada.
3. ¿Se guarda historial diario de pronósticos (escenario B)? ¿Cuántos años de retención?
4. ¿El pronóstico de TPH se calcula por tipo (financiero / no financiero)? Esto duplica `pronosticos_tph`.
5. ¿Spark en modo local o con ejecutores en Kubernetes?
6. Cantidad de usuarios concurrentes del dashboard y de la vista conversacional.
7. Versión de AKS disponible y SKUs de VM permitidos en el banco.
8. ¿Modelo de recuperación y RPO/RTO que exige el banco para SQL Server?
9. Versiones exactas del entorno actual (`pip freeze`) para fijar el `requirements`.
