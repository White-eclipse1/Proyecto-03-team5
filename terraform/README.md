# P2-06 — Un módulo Terraform por capa

Código de infraestructura AWS, sin despliegue. Los roots `environments/dev` y
`environments/prod` consumen los mismos cuatro módulos locales:

| Módulo | Definición mínima |
|---|---|
| `network` | VPC y dos subredes sin IP pública ni rutas a Internet. RDS requiere dos zonas. |
| `compute` | Una EC2 `t3.micro`, disco cifrado, IMDSv2 y security group sin entrada pública. |
| `data` | Una RDS MariaDB `db.t3.micro`, 20 GiB cifrados, backups retenidos 7 días y acceso solo desde compute. RDS administra la contraseña en Secrets Manager. |
| `storage` | Un bucket de artefactos y otro para sus access logs, con nombres generados, cifrado SSE-S3, bloqueo de acceso público y políticas HTTPS-only. |

Cada entorno tiene su propia configuración, etiquetas, nombres y key de estado
S3 (P2-25, explicado abajo). Dev usa `10.10.0.0/16`; prod, `10.20.0.0/16`. Ambos usan por defecto
`us-east-1` y dos zonas distintas. No se agregan tamaños ni servicios adicionales
solo por llamarse prod: esta es una base mínima, no una arquitectura de producción.

Los buckets se generarían con prefijos `mlops-p2-dev-artifacts-` y
`mlops-p2-prod-artifacts-`. Son independientes de `mlops-p2-dvc-cache` y
`mlops-p2-dataset-releases`: no se referencian, importan ni modifican esos recursos.
Tampoco se cambian DVC, MinIO, Docker Compose, el portal o el pipeline Python.

Los access logs se entregan al bucket independiente `${name}-access-logs-...`,
en el prefijo `access-logs/`. Su policy permite únicamente `s3:PutObject` al
servicio `logging.s3.amazonaws.com`, restringido al ARN del bucket origen y a
la cuenta resuelta mediante `aws_caller_identity`, sin identificadores hardcodeados.
Ambos buckets deniegan acciones S3 sobre el bucket y sus objetos cuando
`aws:SecureTransport` es `false`; esa denegación no concede acceso público.
El receptor no genera access logs hacia sí mismo ni hacia el origen, evitando
recursión según las [recomendaciones de AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html).
No se agregan supresiones de análisis ni se modifican los buckets de P2-04.

## Validación

Requisitos: Terraform CLI >= 1.10 y < 2.0, y acceso al registro para descargar
el provider `hashicorp/aws` 6.x. No se necesitan credenciales AWS, sesión SSO
ni una AMI real para estos comandos; no consultan ni despliegan recursos AWS.
Si falta el ejecutable, instala Terraform siguiendo la
[guía oficial](https://developer.hashicorp.com/terraform/install) antes de continuar.

Desde la raíz del repositorio:

```bash
terraform -chdir=terraform fmt -recursive
terraform -chdir=terraform fmt -check -recursive

terraform -chdir=terraform/environments/dev init -backend=false
terraform -chdir=terraform/environments/dev validate

terraform -chdir=terraform/environments/prod init -backend=false
terraform -chdir=terraform/environments/prod validate
```

Ambos roots referencian los cuatro módulos, por lo que `validate` también
comprueba su configuración. No hay un root adicional en `terraform/`.
Los `.terraform.lock.hcl` se generan por entorno durante `init` y deben conservarse
en Git. No se suministran lockfiles inventados ni se versiona `.terraform/`.

La validación estática comprueba sintaxis, referencias y esquema del provider;
no garantiza permisos, cuotas, compatibilidad de versiones ni un despliegue exitoso.
Referencia: [terraform validate](https://developer.hashicorp.com/terraform/cli/commands/validate).

### Autenticación local y límites de onboarding

Cuando una tarea de Terraform requiera consultar AWS y el administrador lo haya
autorizado, Terraform local puede usar el perfil AWS `mlops-p2` mediante la
cadena normal de credenciales del SDK. Ese perfil se obtiene con IAM Identity
Center / SSO; no se deben guardar access keys, contraseñas, sesiones ni tokens
en este repositorio.

GitHub Actions usa OIDC para autenticarse en AWS y es un mecanismo diferente:
la autenticación OIDC de GitHub no autentica automáticamente la Mac de una
persona desarrolladora. No ejecutes `terraform apply` ni `terraform destroy`
como parte del onboarding. No inicialices el backend remoto hasta que el
administrador proporcione y confirme el bucket de state correspondiente.

## Límites del ticket

No ejecutar `apply` ni importar recursos existentes como parte de P2-06.
P2-06 no incluye workflows ni GitHub OIDC (ver P2-07 abajo), VPC Gateway Endpoint (P2-15), NAT,
Internet Gateway, balanceadores, instalación de aplicaciones ni migración de datos.
En particular, la EC2 no tiene conectividad a S3 o Internet ni un mecanismo de
administración remota: la integración operativa corresponde a trabajos posteriores.

Antes de cualquier despliegue futuro habría que seleccionar una AMI real x86_64,
confirmar zonas, versión MariaDB y tamaños admitidos, y revisar conectividad,
permisos y ciclo de vida. RDS usaría una base nueva; no reemplaza MariaDB local.
La política de snapshot final y la conservación del contenido S3 también deben
revisarse antes de una futura eliminación de infraestructura.

Los archivos `terraform.tfvars.example` contienen solo parámetros no sensibles.
No incluyas perfiles personales, access keys, contraseñas o tokens en Terraform.
El provider no fija una identidad: cualquier futura autenticación se resolvería
fuera del código. Los estados, planes y variables locales están ignorados.

## P2-07 — GitHub Actions con OIDC

El root independiente `bootstrap/github-oidc` define un IAM OIDC provider para
`https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`, y
el rol `mlops-p2-github-oidc`. No depende de los módulos de P2-06 ni los despliega.
No se adjuntan políticas de acceso a recursos al rol: este ticket comprueba solo
autenticación. `sts:GetCallerIdentity` no necesita permisos adicionales.

La trust policy permite `sts:AssumeRoleWithWebIdentity` exclusivamente con
`aud = sts.amazonaws.com` y
`sub = repo:karenelizabg/proyecto-fase2-MLOPS:ref:refs/heads/main`.
No permite otros repositorios, ramas, tags, pull requests ni subjects de GitHub
Environments. El job OIDC no declara `environment` para conservar ese subject.

### Validación estática y prueba real

El workflow `.github/workflows/terraform-oidc.yml` valida los roots en PRs
hacia `main`, pushes a `main` y ejecuciones manuales. El job de validación solo
tiene `contents: read`; no intenta asumir un rol ni necesita credenciales AWS.
Las Actions están fijadas a commits verificados de sus repositorios oficiales.

Además de los comandos de P2-06, valida el nuevo root con:

```bash
terraform -chdir=terraform fmt -recursive
terraform -chdir=terraform fmt -check -recursive
terraform -chdir=terraform/bootstrap/github-oidc init -backend=false
terraform -chdir=terraform/bootstrap/github-oidc validate
```

El job OIDC depende de la validación y solo corre en el repositorio exacto, desde
`main`, por push o ejecución manual. Tiene `contents: read` e `id-token: write`,
utiliza `aws-actions/configure-aws-credentials` con
`role-to-assume: ${{ vars.AWS_ROLE_ARN }}` y una sesión de 15 minutos. La prueba
STS descarta su salida y no imprime identificadores de identidad.

La validación del código no demuestra una autenticación real. Esta última queda
pendiente hasta que una persona autorizada cree el provider/rol fuera de esta
tarea y registre el output `role_arn` como variable de repositorio `AWS_ROLE_ARN`
en GitHub. El ARN no es una credencial y no necesita escribirse en los archivos.
Si `AWS_ROLE_ARN` no existe, el workflow emite un warning y omite la prueba
OIDC; no falla por la ausencia de esa variable. Cuando la variable está
configurada, el job intenta asumir el rol y verifica STS. Los PRs no ejecutan
ese job de `main` y no fallan por falta de autorización de su rama.

### Provider existente y seguridad

Antes de una futura creación, comprueba si la cuenta ya tiene el provider GitHub.
Si existe, suministra su ARN mediante `TF_VAR_existing_oidc_provider_arn` o un
archivo de variables local ignorado. Verifica que pertenece a la cuenta destino
y que tiene la URL y audience indicadas. En ese modo Terraform no crea, importa
ni modifica ese provider; únicamente lo referencia en la trust policy del rol.
No cambies entre modos sobre un estado que ya gestione el provider sin revisar
su ciclo de vida: podría proponer eliminarlo. No se hace esa transición aquí.

La creación inicial requiere una identidad humana autorizada externa al proyecto;
el workflow no puede crear su propio rol de arranque. No se guardan perfiles
personales, credenciales estáticas, tokens ni identificadores de cuenta en código.
No se ejecutan `plan`, `apply` ni importaciones como parte de esta implementación.
No se agregan permisos S3, EC2, RDS, IAM o Secrets Manager, ni recursos de P2-15.

## P2-15 — Buckets S3 versionados + VPC Gateway Endpoint

Extiende los módulos `storage` y `network` de P2-06; no agrega un root nuevo,
por lo que ya queda cubierto por la validación existente (`environments/dev`
y `environments/prod` consumen ambos módulos).

`modules/storage` agrega dos buckets independientes de `this`
(el bucket de artefactos de P2-06): `dvc_cache` y `dataset_releases`, con
`aws_s3_bucket_versioning` habilitado y el mismo baseline de seguridad que
ya usa el módulo (bloqueo de acceso público, cifrado SSE-S3, deny
HTTPS-only, access logs entregados al mismo bucket `logs` de P2-06, cada
uno bajo su propio prefijo `access-logs/dvc-cache/` y
`access-logs/dataset-releases/`). La policy de `logs` se amplió para
permitir la entrega de logs desde estos dos buckets además de `this`.
Igual que en P2-06, los nombres se generan con `bucket_prefix` (p. ej.
`mlops-p2-dev-dvc-cache-<sufijo>`): son independientes de
`mlops-p2-dvc-cache` y `mlops-p2-dataset-releases` (los buckets reales de
P2-04); no se referencian, importan ni modifican.

`modules/network` agrega un `aws_vpc_endpoint` tipo Gateway para S3,
asociado a la route table por defecto de la VPC — la única que existe,
porque P2-06 no crea route tables propias (las subredes no tienen rutas
públicas). El nombre del servicio se resuelve con `data "aws_region"
"current"`, sin región hardcodeada.

No se ejecuta `apply` ni se modifica ningún recurso real de AWS como parte
de este ticket, igual que P2-06 y P2-07.

## P2-25 — Estado remoto S3 con locking

`bootstrap/remote-state` define un bucket dedicado al estado de Terraform y
su receptor de access logs, independientes de los buckets DVC/dataset y de los módulos de los
entornos. Usa el prefijo configurable `bucket_prefix` (`mlops-p2-tfstate-` por
defecto); el provider agrega un sufijo para generar un nombre único. La región
se configura con `region` (`us-east-1` por defecto). Los outputs son
`state_bucket_name` y `state_bucket_region`.

El bucket tiene versionado, cifrado SSE-S3 AES256, bloqueo completo de acceso
público y denegación HTTPS-only, sin permisos públicos. `force_destroy=false`
y `prevent_destroy=true` protegen frente a eliminaciones accidentales mediante
Terraform mientras se conserve la configuración; no sustituyen controles IAM
ni copias de seguridad. No se crea DynamoDB ni infraestructura de otros tickets.

### Access logging y Sonar S6258

`aws_s3_bucket_logging.state` registra accesos al bucket de state en un receptor
dedicado `aws_s3_bucket.logs`, creado por este mismo bootstrap en la misma cuenta
y región. Su nombre usa `bucket_prefix="mlops-p2-tfstate-logs-"` con sufijo generado.
Tiene SSE-S3 AES256, bloqueo completo de acceso público y política HTTPS-only.
La única concesión de entrega permite `s3:PutObject` a `logging.s3.amazonaws.com`
sobre `access-logs/*`, condicionada al ARN del bucket de state y a la cuenta
obtenida mediante `aws_caller_identity`. No hay identificadores de cuenta ni
credenciales hardcodeados. La configuración de logging depende explícitamente
de la policy, cifrado y bloqueo público del receptor.

El receptor **no tiene server access logging**, siguiendo la
[recomendación de AWS para buckets destino](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-server-access-logging.html).
No envía logs a sí mismo ni al bucket de state y no se crea un tercer bucket.
Si Sonar S6258 señala específicamente `aws_s3_bucket.logs` en este root, revisar
ese issue individual: el bucket es deliberadamente un destino de logs protegido,
no un bucket de datos que haya omitido auditoría accidentalmente. La justificación
debe identificar este recurso, citar AWS, explicar la ausencia de logging recursivo
y verificar las protecciones y restricciones de entrega anteriores.

Tras esa revisión, una persona autorizada puede resolver **solo ese issue** como
`False positive` por el contexto de destino de logs; si la política del equipo
lo trata como riesgo aceptado, documentar esa decisión con el estado permitido.
No desactivar la regla, excluir archivos/directorios ni añadir `NOSONAR`. Un
comentario explicativo no resuelve automáticamente un issue en Sonar. Esta
configuración no cambia estados de issues; debe comprobarse el próximo análisis.

### Bootstrap y orden de operación

Primero una persona autorizada debe provisionar el bucket desde el root
`bootstrap/remote-state`, en una operación futura separada. **Esta entrega no
crea recursos AWS ni ejecuta apply, plan o migraciones.** El bootstrap usa el
backend local implícito: no puede depender del bucket que todavía va a crear.
Su state inicial permanece local, ignorado por Git, y debe custodiarse y
respaldarse de forma segura. No borrarlo ni recrear el bootstrap desde un clon
sin recuperar su estado; ignorarlo no equivale a respaldarlo.

Solo cuando el bucket exista y esté configurado se inicializan dev y prod con
él. No se obtiene el bucket desde `module.storage` ni desde variables/outputs
del propio root: Terraform inicializa el backend antes de evaluar esos recursos.
`bootstrap/github-oidc` conserva su backend local y su comportamiento de P2-07.

### Configuración parcial e inicialización futura

Los `backend.tf` fijan `encrypt=true`, `use_lockfile=true` y keys diferentes:

| Root | Key en el bucket compartido |
|---|---|
| `environments/dev` | `environments/dev/terraform.tfstate` |
| `environments/prod` | `environments/prod/terraform.tfstate` |

Bucket y región se suministran con `-backend-config`. No son variables Terraform
del entorno; la región del backend debe ser la del bucket, aunque la región de
los recursos de un entorno sea distinta. Después del aprovisionamiento autorizado,
desde la raíz del repositorio, para un root sin estado local previo:

```bash
state_bucket_name="$(terraform -chdir=terraform/bootstrap/remote-state output -raw state_bucket_name)"
state_bucket_region="$(terraform -chdir=terraform/bootstrap/remote-state output -raw state_bucket_region)"

terraform -chdir=terraform/environments/dev init \
  -backend-config="bucket=$state_bucket_name" \
  -backend-config="region=$state_bucket_region"

terraform -chdir=terraform/environments/prod init \
  -backend-config="bucket=$state_bucket_name" \
  -backend-config="region=$state_bucket_region"
```

Estas inicializaciones **sí contactan S3** y no forman parte de la validación
estática. En otro equipo sin el state del bootstrap, una persona autorizada
proporciona su nombre y región de salida y se asignan directamente esas dos
variables de shell. No ejecutar de nuevo el bootstrap para descubrir el nombre.

La identidad se resuelve externamente mediante la cadena de credenciales AWS
(por ejemplo sesión SSO autorizada). No poner claves, tokens ni perfiles
personales en `.tf` ni en argumentos de configuración del backend. Terraform
guarda metadatos de backend bajo `.terraform/`, que permanece ignorado.

### Locking y permisos

`use_lockfile=true` requiere Terraform >= 1.10. El backend utiliza un objeto
`<key>.tflock` para coordinar operaciones que bloquean el estado de esa key.
Dev y prod tienen locks independientes. No usar `-lock=false`; ante un lock
abandonado, verificar primero que no haya otra operación activa antes de
considerar un desbloqueo administrativo.

El operador necesita `s3:ListBucket` limitado a los prefijos correspondientes,
`s3:GetObject`/`s3:PutObject` sobre el state y
`s3:GetObject`/`s3:PutObject`/`s3:DeleteObject` sobre su `.tflock`. No necesita
eliminar el objeto state. El rol OIDC actual solo prueba identidad y no recibe
permisos S3 en este ticket; autenticarlo no demuestra acceso al backend.

`.terraform.lock.hcl` fija providers y checksums: se mantiene versionado y no
es el archivo de locking del state. El nuevo root conserva la selección AWS
6.64.0 y los checksums existentes, sin actualizar el provider.

### Estado local existente

Antes de una migración, detener operaciones concurrentes, respaldar el state
fuera de Git, verificar bucket/key/identidad y comprobar si el destino ya tiene
estado. No sobrescribir un destino ocupado. Solo tras revisar y autorizar la
migración, usar `terraform init -migrate-state` en el root correspondiente con
los mismos argumentos `-backend-config` y revisar su confirmación. No automatizar
`-force-copy`; `-reconfigure` no sustituye la migración de un state existente.
Conservar el respaldo hasta verificar el destino. No se migra nada en esta entrega.

Nunca subir `.tfstate`, backups, `.terraform/` ni planes a Git. Las exclusiones
globales complementan `terraform/.gitignore`, conservando los lockfiles de providers.

### Validación sin AWS

```bash
terraform fmt -check -recursive terraform
for root in environments/dev environments/prod bootstrap/github-oidc bootstrap/remote-state; do
  terraform -chdir="terraform/$root" init -backend=false -input=false -lockfile=readonly
  terraform -chdir="terraform/$root" validate
done
git diff --check
git ls-files | grep -E '\.tfstate|\.terraform/'
```

El último comando no debe imprimir nada (`grep` devuelve 1 si no hay coincidencias).
El workflow incluye los cuatro roots, sin activar los backends. Estas pruebas
pueden descargar providers, pero no demuestran existencia del bucket, permisos,
migración ni locking concurrente real. Esas comprobaciones quedan para una
operación AWS autorizada posterior.
