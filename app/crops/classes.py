"""ML-01 — clases oficiales del clasificador del Proyecto 3.

`CLASS_NAMES` es la única fuente de verdad: dog y cat, en ese orden. El
índice de cada clase (`CLASS_TO_INDEX`) es el que usará ML-02/ML-03 como
etiqueta del modelo, así que cambiar este orden invalida checkpoints
previos. Los `category_id` del COCO de P2 (dog=3, cat=4 en v0.1.1) no se
fijan aquí: se resuelven por nombre contra las categorías del release.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

CLASS_NAMES: tuple[str, ...] = ("dog", "cat")
CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(CLASS_NAMES)}
)


def resolve_category_classes(categories: Iterable[Mapping]) -> dict[int, str]:
    """Devuelve `{category_id: clase}` para las categorías COCO dog y cat.

    Otras categorías del release (p. ej. `person`) se ignoran aquí y sus
    anotaciones se registran como excluidas en la extracción. Un release sin
    dog o cat, o con una clase repetida bajo dos ids, se rechaza.
    """
    resolved: dict[int, str] = {}
    for category in categories:
        name = category["name"]
        if name not in CLASS_TO_INDEX:
            continue
        if name in resolved.values():
            raise ValueError(f"La clase '{name}' aparece con más de un category_id")
        resolved[category["id"]] = name

    missing = [name for name in CLASS_NAMES if name not in resolved.values()]
    if missing:
        raise ValueError(f"El release no declara las clases oficiales: {missing}")
    return resolved
