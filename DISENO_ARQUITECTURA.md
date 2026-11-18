# Diseño de arquitectura – BaNCS Forecast

Diagramas en Mermaid (se visualizan en VS Code con la vista previa de Markdown, en GitHub y en Azure DevOps). Descripción de cada componente en [ARQUITECTURA.md](./ARQUITECTURA.md).

## 1. Arquitectura general

```mermaid
flowchart LR
  subgraph SRC[Fuentes BaNCS]
    S1[Transacciones]
    S2[Colas]
    S3[Procesos críticos]
    S4[Reposting]
  end

  subgraph ONPREM[Servidores del banco - On-premise]
    direction TB
    AF[Airflow<br/>orquestación]
    subgraph MED[Spark + Delta Lake - Medallón]
      B[(BRONZE<br/>crudo)]
      SV[(SILVER<br/>limpio y validado)]
      G[(GOLD<br/>agregados + predicciones)]
    end
    ML[Modelo de pronóstico<br/>Spark MLlib / Prophet]
    DQ[Calidad de datos<br/>Great Expectations / Deequ]
    Q[Capa de consulta<br/>Thrift / Trino / DuckDB]
    API[API FastAPI<br/>RBAC + auditoría]
    WEB[Nginx / IIS<br/>HTTPS + proxy]
  end

  subgraph AZ[Microsoft Azure]
    ENTRA[Entra ID<br/>SSO + grupos]
  end

  USER[Usuario autorizado<br/>Intranet / VPN]

  SRC --> AF --> B --> SV --> G
  DQ -. valida .-> SV
  DQ -. valida .-> G
  G --> ML --> G
  G --> Q --> API --> WEB --> USER
  USER -. login OIDC .-> ENTRA
  API -. valida token y grupos .-> ENTRA
```

## 2. Flujo del medallón

```mermaid
flowchart LR
  F[Archivos / BD / logs] -->|ingesta + fecha_ingesta, archivo_origen| B[(Bronze)]
  B -->|tipado, deduplicación,<br/>ajuste día hábil D+1,<br/>exclusión de copias/pruebas| SV[(Silver)]
  SV -->|filas rechazadas| R[(Tabla de excluidas)]
  SV -->|agregación por hora/día/proceso| G[(Gold)]
  G -->|pronóstico + intervalo| G
```

## 3. Autenticación y autorización por rol

```mermaid
sequenceDiagram
  actor U as Usuario
  participant W as Nginx/IIS
  participant E as Entra ID / AD
  participant A as API (RBAC)
  participant G as Gold (Delta)

  U->>W: Abre el sitio (HTTPS)
  W->>E: Redirige a login (OIDC)
  E-->>U: Autentica (MFA) y emite token con grupos
  U->>W: Solicitud con token
  W->>A: Proxy de la solicitud
  A->>A: Valida token, grupo -> rol
  alt Sin grupo autorizado
    A-->>U: 403 Acceso denegado
  else Rol válido
    A->>G: Consulta solo lo permitido para el rol
    G-->>A: Datos
    A-->>U: Perfil del rol + datos
    A->>A: Registra auditoría
  end
```

## 4. Pantallas por rol

```mermaid
flowchart TB
  A[API: perfil del usuario] --> E[Ejecutivo<br/>Resumen, KPI, pronóstico agregado]
  A --> O[Operaciones<br/>Colas, Procesos críticos, Reposting]
  A --> AN[Analista<br/>Todas las pestañas, rangos, exportar]
  A --> AD[Administrador<br/>Todo + configuración + auditoría]
```

## 5. Zonas de red

```mermaid
flowchart LR
  U[Usuarios<br/>Intranet / VPN] -->|443| DMZ
  subgraph DMZ[Zona de aplicación]
    WEB[Nginx / IIS]
    API[API]
  end
  subgraph INT[Zona interna de datos - sin acceso de usuarios]
    SP[Spark]
    ST[(Delta en HDFS / MinIO / NAS)]
    AF[Airflow]
  end
  subgraph OPS[Operación]
    MON[Prometheus + Grafana]
    LOG[Logs / SIEM]
    VLT[Vault]
  end
  WEB --> API --> SP --> ST
  AF --> SP
  MON -.-> DMZ
  MON -.-> INT
  DMZ -.-> LOG
  INT -.-> LOG
  API -.-> VLT
  AF -.-> VLT
```

## 6. Estructura de repositorio sugerida

```
DayTransactionsPredictor/
├─ ARQUITECTURA.md
├─ DISENO_ARQUITECTURA.md
├─ dashboard/            # frontend (consume la API)
├─ api/                  # FastAPI: auth, RBAC, endpoints, auditoría
├─ spark/
│  ├─ bronze/            # ingesta
│  ├─ silver/            # limpieza y reglas
│  ├─ gold/              # agregados
│  └─ forecast/          # modelo de pronóstico
├─ airflow/dags/         # un DAG por dominio
├─ quality/              # expectativas de calidad
├─ deploy/               # nginx/IIS, systemd/Docker, CI/CD
└─ tests/
```
