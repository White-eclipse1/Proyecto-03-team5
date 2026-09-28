"""Tier 3 — la compuerta de calidad.

Lee `policies/quality.yaml`, compara contra los 5 reportes de `analyzers` y
decide PASA / BLOQUEA. Si bloquea, las muestras regresan a la cola de
re-anotación (tier 1), no siguen de largo en el pipeline.
"""
