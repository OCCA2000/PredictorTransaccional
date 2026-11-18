# Arquitectura – BaNCS Forecast (DayTransactionsPredictor)

> Estado: **propuesta para validación** con Infraestructura, Seguridad y Arquitectura del banco.
> Principio rector: **on-premise primero**; Azure solo para identidad (y opcionalmente respaldo/monitoreo).

## 1. Objetivo

Publicar el dashboard de pronóstico de BaNCS (TXN financieras y no financieras, Colas, Procesos Críticos y Reposting) en un sitio web interno, con:

- Procesamiento de datos en un **servidor local** del banco usando **Apache Spark** y **arquitectura medallón** (Bronze / Silver / Gold).
- Acceso restringido a personas autorizadas, no a toda la organización.
- **Pantalla distinta según el rol** del usuario.

## 2. Vista general

```
Fuentes BANCS ─► Ingesta (Airflow) ─► BRONZE ─► SILVER ─► GOLD ─► Forecast (Spark MLlib/Prophet)
                                                              │
                                                  API (FastAPI) + RBAC ◄── Entra ID / AD (SSO)
                                                              │
                                                  Nginx/IIS (HTTPS) ─► Intranet/VPN ─► Navegador
```

Diagramas detallados en [DISENO_ARQUITECTURA.md](./DISENO_ARQUITECTURA.md).

## 3. Componentes: qué se usa y para qué

### 3.1 Datos y procesamiento

| Componente | Para qué se usa | Notas |
|---|---|---|
| **Apache Spark** | Motor de procesamiento distribuido para transformar Bronze→Silver→Gold y entrenar/ejecutar el pronóstico. | Standalone al inicio; YARN o Kubernetes si crece. |
| **Delta Lake** | Formato de tablas del medallón: transacciones ACID, historial de versiones (time travel), evolución de esquema, `MERGE` para deduplicar. | Sobre almacenamiento local. |
| **Almacenamiento local (HDFS, MinIO o NAS/SAN con RAID)** | Guardar las tablas Delta. | MinIO da API compatible S3; decidir según infraestructura existente. |
| **Apache Airflow** | Orquestar y programar los jobs (ingesta, capas, modelo), con reintentos, dependencias y alertas. | Un DAG por dominio: TXN, colas, procesos, reposting. |
| **Great Expectations / Deequ** | Controles de calidad de datos entre capas (nulos, rangos, duplicados, completitud por día). | Un fallo detiene la promoción a la capa siguiente. |
| **PySpark** | Lenguaje de los jobs de transformación. | Código versionado y con pruebas. |
| **Spark MLlib / Prophet** | Modelo de pronóstico y su intervalo de confianza. | El modelo concreto se mantiene el actual; esto define dónde corre. |
| **Spark Thrift Server / Trino / DuckDB sobre Delta** | Capa de consulta de Gold para la API. | Elegir según concurrencia esperada; DuckDB basta con pocos usuarios. |

### 3.2 Capas del medallón

| Capa | Contenido | Reglas |
|---|---|---|
| **Bronze** | Datos crudos tal como llegan (TXN, colas, procesos, reposting), con `fecha_ingesta` y `archivo_origen`. | Inmutable, solo se agrega. Permite reprocesar. |
| **Silver** | Datos limpios, tipados y conformados. | Reglas hoy en el dashboard: reposting del día hábil D corre en D+1 y se ajusta al día hábil anterior; se excluyen copias de otro día, depuraciones/pruebas, corridas de fin de semana sin confirmar y volúmenes repetidos; las excluidas se guardan en una tabla de rechazos (equivalente a `reposting_excluidas.csv`). Deduplicación y calendario de días hábiles. |
| **Gold** | Tablas listas para consumo: TPH por hora, colas, procesos críticos, reposting, y predicciones con intervalo. | El dashboard **solo** lee Gold. |

### 3.3 Aplicación web

| Componente | Para qué se usa | Notas |
|---|---|---|
| **Nginx (Linux) o IIS (Windows)** | Servidor web: entrega el HTML/JS del dashboard por **HTTPS** y actúa como proxy inverso hacia la API. | Certificado emitido por la CA interna del banco. |
| **API (FastAPI)** | Punto único de acceso a los datos. Valida identidad y rol, y devuelve solo lo permitido. | El navegador nunca consulta Spark/Delta directamente. |
| **Dashboard (HTML/JS)** | Interfaz. Hoy es un solo archivo con datos embebidos; pasa a consumir la API. | La UI por rol se arma con el perfil que entrega la API. |

### 3.4 Identidad, acceso y roles

| Componente | Para qué se usa | Notas |
|---|---|---|
| **Microsoft Entra ID (Azure AD)** | Inicio de sesión único (SSO) con OIDC. Es el único uso de Azure en el núcleo. | Alternativa 100 % local: AD FS o Active Directory (Kerberos/LDAP). |
| **Grupos de seguridad de AD** | Un grupo por rol; solo sus miembros acceden. | Altas y bajas las gestiona el equipo de accesos, no la aplicación. |
| **RBAC en la API** | Decide qué pestañas y datos ve cada rol, validado en el servidor en cada petición. | Ocultar elementos en el HTML es solo cosmético. |
| **Auditoría** | Registrar quién accedió y qué consultó. | Almacenar en tabla propia / SIEM del banco. |

Roles propuestos (a confirmar con el negocio):

| Rol | Ve |
|---|---|
| Ejecutivo | Resumen, indicadores clave y pronóstico agregado. |
| Operaciones / Monitoreo | Colas, procesos críticos y reposting en detalle. |
| Analista | Todas las pestañas con detalle, rangos de fecha y exportación. |
| Administrador | Todo lo anterior, gestión de configuración y consulta de auditoría. |

### 3.5 Seguridad y operación

| Componente | Para qué se usa |
|---|---|
| **Segmentación de red / firewall** | Spark y almacenamiento en zona interna sin acceso de usuarios; solo Nginx/IIS y la API en la intranet. |
| **Cifrado** (TLS en tránsito, disco/almacenamiento en reposo) | Proteger datos operativos del banco. |
| **HashiCorp Vault o equivalente** | Guardar secretos y credenciales fuera del código. |
| **Prometheus + Grafana** | Monitoreo de servidores, Spark, Airflow y API; alertas. |
| **Logs centralizados (ELK/Loki o SIEM del banco)** | Diagnóstico y trazabilidad. |
| **Backups** | Respaldo de Gold, configuración y metadatos de Airflow. |
| **Git + CI/CD** | Versionar código y desplegar de dev a QA y a prod. |

### 3.6 Rol de Azure (opcional)

| Servicio | Uso |
|---|---|
| Entra ID | **Recomendado**: SSO y grupos. |
| Azure Key Vault | Alternativa a Vault para secretos. |
| Azure Monitor / Arc | Monitoreo y gestión híbrida de los servidores locales. |
| Azure Backup / Blob (frío) | Copia de respaldo externa. |

Los datos y el procesamiento permanecen on-premise. Si Seguridad no permite ningún servicio externo, se usa AD FS/AD local.

## 4. Flujo de datos

1. Los datos de BaNCS (logs, BD, archivos) llegan al servidor y Airflow los carga a **Bronze**.
2. Spark limpia y conforma a **Silver**, aplicando reglas y controles de calidad.
3. Spark agrega a **Gold** y el modelo escribe las predicciones.
4. El usuario abre el sitio, Entra ID autentica y la API lee los grupos de AD para determinar el rol.
5. La API consulta Gold, filtra según el rol y responde; el dashboard dibuja la pantalla de ese rol.

## 5. Dimensionamiento inicial (a validar con volumen real)

| Servidor | Especificación orientativa |
|---|---|
| Spark + Delta + Airflow | 16+ cores, 64+ GB RAM, SSD/NVMe con RAID. |
| Web + API | 4–8 cores, 16 GB RAM. |
| Alta disponibilidad (opcional) | Segundo nodo Spark y segundo servidor web con balanceo. |

## 6. Ambientes

Dev, QA y Prod, con datos enmascarados fuera de Prod.

## 7. Fases

1. **Datos:** servidor Spark/Delta y carga Bronze/Silver/Gold de los datos actuales.
2. **Aplicación:** API con SSO y roles; dashboard conectado a la API.
3. **Operación:** Airflow completo, calidad de datos, monitoreo, auditoría y backups.
4. **Piloto** con un grupo reducido, luego producción.

## 8. Decisiones pendientes

- Sistema operativo del servidor (Linux o Windows) y servidor web (Nginx o IIS).
- Almacenamiento: HDFS, MinIO o NAS/SAN.
- Frecuencia de actualización: diaria, por hora o casi en tiempo real (esto puede exigir Spark Structured Streaming).
- Volumen diario de datos y fuentes reales (archivos CSV o conexión directa a BD/logs).
- Identidad: Entra ID o AD FS/AD local.
- Lista definitiva de roles y qué ve cada uno.
- Requisitos regulatorios del banco (retención de datos, auditoría, aprobación de proveedores).
