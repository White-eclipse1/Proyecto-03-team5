"""Tier 2 — analizadores de calidad (objetos pequeños, desbalance, duplicados,
cajas inválidas, sesgo espacial).

Reglas de la capa (ver `app/tests/test_architecture.py` para la lista exacta
de patrones prohibidos, verificados también por CI):

- Ningún módulo de este paquete lee variables de entorno directamente, ni
  crea clientes de red o de base de datos por su cuenta.
- Cada analizador es una función pura: entra el COCO ya cargado (o la
  estructura que corresponda) y sale un `AnalyzerResult` (ver `base.py`).
- Cualquier acceso a MinIO/S3/MariaDB pasa por el paquete `storage`,
  inyectado por quien orquesta el pipeline, nunca importado aquí.
"""
