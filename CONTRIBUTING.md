# Cómo contribuir

Guía corta para trabajar en equipo sin pisarnos. Lo que aquí se pide lo revisa el CI
(`.github/workflows/`), así que si algo se pone en rojo, casi siempre el mensaje dice qué
corregir.

## Flujo en cinco pasos

1. **Un ticket = una rama = un PR.** No mezcles tickets en una misma rama salvo que el
   propio ticket los agrupe (por ejemplo `P2-26 P2-27`).
2. **Parte de `main` actualizado:**
   ```bash
   git fetch origin
   git checkout -b p2-NN-descripcion origin/main
   ```
3. **Haz commits pequeños** con el formato de abajo.
4. **Corre en local lo mismo que el CI** (tabla más abajo) antes de subir.
5. **Abre el PR** con el título en formato y la plantilla completa. Si el CI está en rojo,
   no se mergea.

## Ramas

`p2-NN-descripcion`, todo en minúsculas y con guiones.

| Caso | Ejemplo |
|---|---|
| Un ticket | `p2-52-copilot-llm-client` |
| Varios tickets | `p2-22-23-24-quality-gate-e2e`, `p2-26-p2-27-anotacion-lotes-7-8` |
| Sin ticket | `fix/annotation-id-collisions`, `chore/gitignore-node-modules` |

Los prefijos válidos para ramas sin ticket son `feat`, `fix`, `test`, `chore`, `docs`,
`refactor`, `ci`, `build`, `perf` y `style`.

## Commits

```
P2-NN: qué cambia, en una línea
fix: qué se arregla, en una línea
```

- Formato `P2-NN: descripción` cuando hay ticket, o `tipo: descripción` cuando no
  (mismos tipos que en las ramas).
- Primera línea de unos 72 caracteres, sin punto final. Español o inglés, como prefieras.
- Si hace falta, deja una línea en blanco y explica **por qué** en el cuerpo: el qué ya
  lo dice el diff.
- Un commit = un cambio con sentido. Evita `wip`, `arreglos` o `cambios varios`.
- **Nunca** subas `.env`, claves, credenciales de AWS ni datos: `.env` está ignorado y los
  datos van por DVC.

Esta guía no se verifica commit por commit; sí se verifica el nombre de la rama y el título
del PR (siguiente sección).

## Pull requests

- **Título** con el mismo formato que un commit: `P2-52: Copilot con cliente LLM`,
  `P2-22/23/24: compuerta de calidad`, `P2-26 P2-27: lotes 7 y 8` o
  `fix: main tiene un test roto`. Ojo con `P2-52 - texto`, `P2 52 texto` o `P2-52 texto`:
  no cumplen.
- **Descripción:** completa la plantilla. Di qué probaste **y qué no pudiste probar**.
- **Cierra el issue** con `Cierra #NN` en la descripción.
- Si tu PR **depende de otro sin mergear**, apúntalo a la rama de ese ticket en vez de
  `main`, y cuando el padre se mergee, reapunta el tuyo a `main`.
- Antes de mergear, actualiza tu rama con `main` si otro PR entró mientras trabajabas, y
  espera a que el CI vuelva a pasar. Dos PR que pasan por separado pueden romper `main`
  juntos: pasó con #120 y #121 (ver #131).

Si el título o la rama no cumplen, el job **Nombre de rama y título del PR** falla. Para el
título basta editarlo en GitHub: el job se vuelve a correr solo. Para comprobarlo antes de
abrir el PR:

```bash
bash .github/scripts/check-naming.sh "$(git branch --show-current)" "P2-NN: mi título"
```

## Qué corre el CI

Se ejecuta en cada PR (contra cualquier rama base) y en cada push a `main`. Estos son los
mismos comandos que puedes correr en tu máquina:

| Paquete | Comandos (desde su carpeta) |
|---|---|
| `app/` (Python) | `uv sync --locked --no-build` · `uv run --locked --no-build ruff check .` · `uv run --locked --no-build ruff format --check .` · `uv run --locked --no-build pytest -q` |
| `backend/` | `npm ci --ignore-scripts` · `npm run lint` · `npm run typecheck` · `npm test` · `npm run build` |
| `frontend/` | `npm ci --ignore-scripts` · `npm run lint` · `npm run typecheck` · `npm test` · `npm run build` |
| `docker-compose.yml` | `docker compose config --quiet` (con `MINIO_ROOT_USER` y `MINIO_ROOT_PASSWORD` definidas) |

`--no-build` (uv) y `--ignore-scripts` (npm) evitan ejecutar scripts de instalación o de
compilación de dependencias de terceros: lo pide SonarCloud y `app/Dockerfile` ya lo hacía. Si
una dependencia nueva solo se publica como código fuente, o necesita un script de instalación,
el CI fallará a propósito para que el equipo lo revise y decida.

Para arreglar de golpe lo que Ruff o Biome pueden corregir solos: `uv run ruff check --fix .`
y `uv run ruff format .` en `app/`; `npm run lint:fix` en `backend/` y `frontend/`.

`CI OK` es el check que resume a todos los demás. Si un job falla, `CI OK` también falla.

### Ruff

La configuración vive en `app/pyproject.toml`. Reglas activas: `E`, `F` (errores y
pyflakes), `I` (orden de imports), `B` (bugbear: bugs probables), `SIM` (simplificaciones),
`C4` (comprensiones y literales) y `RUF` (reglas propias de Ruff). Si de verdad necesitas
silenciar una regla en una línea, usa `# noqa: CODIGO` con el motivo al lado; Ruff (`RUF100`)
avisa cuando un `noqa` ya no hace falta.

### Los reportes commiteados también se validan

`app/tests/test_committed_reports.py` comprueba que los JSON reales de `reports/`
(`quality.json`, `projections.json`, `versions.json` y cada release) cumplan los contratos y
que la versión de cada release coincida con el catálogo. Si regeneras un reporte y lo
commiteas roto, este test lo detecta antes del merge.

## Finales de línea en Windows

El repo guarda LF y `.gitattributes` pide LF también en tu carpeta de trabajo. Si tu copia
todavía tiene archivos con CRLF de antes (Biome lo notará con `npm run lint` como un error de
formato), guarda o commitea tu trabajo y ejecuta:

```bash
git add --renormalize .
git status   # no debería mostrar cambios de contenido
```

Y para que la carpeta de trabajo quede en LF, con todo commiteado o guardado:

```bash
git rm --cached -r . && git reset --hard
```

Este último comando **descarta cambios sin commitear**: no lo uses si no has guardado tu
trabajo.

## Para quien administra el repositorio

Estos ajustes se hacen en *Settings → Branches → Branch protection rules* para `main` y
requieren permisos de administración; el código por sí solo no puede activarlos:

- **Require a pull request before merging.**
- **Require status checks to pass**, marcando `CI OK`. Opcionalmente también
  `Nombre de rama y título del PR`.
- **Require branches to be up to date before merging.** Es la que evita el caso de #131:
  obliga a que cada PR se pruebe contra el `main` actual antes de entrar.

Mientras eso no esté activado, los checks en rojo avisan pero no bloquean el botón de merge.
