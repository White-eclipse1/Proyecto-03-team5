"""`python -m tracking.check`: ¿este contenedor alcanza al servidor MLflow?

Sirve para comprobar la conexión desde el worker o el servicio `app`:

    docker compose run --rm app python -m tracking.check
"""

import sys

from tracking.client import TrackingServerUnavailableError, check_server, tracking_client


def main() -> int:
    try:
        uri = check_server()
    except TrackingServerUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    experiments = tracking_client().search_experiments(max_results=100)
    print(f"MLflow OK en {uri} ({len(experiments)} experimentos visibles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
