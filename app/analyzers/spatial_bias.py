"""P2-29: sesgo espacial de las cajas COCO [x, y, width, height], sin I/O."""

from statistics import fmean, median, pstdev, quantiles

from pydantic import BaseModel, ConfigDict, Field

from analyzers.base import AnalyzerResult


class SpatialBiasConfig(BaseModel):
    """Sin defaults: el umbral se suministra desde la configuración de políticas."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    min_std_dev: float = Field(ge=0, le=0.5)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{percentile}": 0.0 for percentile in (25, 50, 75, 90, 95)}
    if len(values) == 1:
        return {f"p{percentile}": values[0] for percentile in (25, 50, 75, 90, 95)}
    values_by_percentile = quantiles(values, n=100, method="inclusive")
    return {
        f"p{percentile}": values_by_percentile[percentile - 1]
        for percentile in (25, 50, 75, 90, 95)
    }


def analyze_spatial_bias(coco: dict, config: SpatialBiasConfig) -> AnalyzerResult:
    """Centro normalizado de cada bbox (0-1 en cada eje, relativo a su imagen).

    El sesgo se mide como dispersión: si los anotadores siempre centran el
    objeto, los centros se acumulan cerca de (0.5, 0.5) y la desviación
    estándar por eje es baja. `metric_value` es la menor de las dos
    desviaciones (el eje con menos dispersión es el más sesgado); `passed`
    exige que ambos ejes superen `min_std_dev`. Una distribución uniforme en
    [0, 1] tiene desviación ≈0.289, así que un umbral típico va bien por
    debajo de eso. No se usan `area` ni tamaño de caja, solo posición del
    centro. Se cuentan todas las anotaciones, incluidas `iscrowd`.
    """
    images = {image["id"]: image for image in coco["images"]}
    centers_x: list[float] = []
    centers_y: list[float] = []

    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        centers_x.append((x + width / 2) / image["width"])
        centers_y.append((y + height / 2) / image["height"])

    if len(centers_x) < 2:
        std_x = std_y = 0.0
    else:
        std_x = pstdev(centers_x)
        std_y = pstdev(centers_y)
    metric_value = min(std_x, std_y)
    median_x = median(centers_x) if centers_x else 0.0
    median_y = median(centers_y) if centers_y else 0.0

    return AnalyzerResult(
        check_name="spatial_bias",
        passed=metric_value >= config.min_std_dev,
        metric_value=metric_value,
        details={
            "total_boxes": len(centers_x),
            "mean_center_x": fmean(centers_x) if centers_x else 0.0,
            "mean_center_y": fmean(centers_y) if centers_y else 0.0,
            "median_center_x": median_x,
            "median_center_y": median_y,
            "std_center_x": std_x,
            "std_center_y": std_y,
            "center_x_percentiles": _percentiles(centers_x),
            "center_y_percentiles": _percentiles(centers_y),
        },
    )
