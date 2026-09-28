from pydantic import BaseModel


class AnalyzerResult(BaseModel):
    """Contrato de salida común a los 5 analizadores de calidad (Tier 2)."""

    check_name: str
    passed: bool
    metric_value: float
    details: dict = {}
