@SPEC-COCO-UI-001
Feature: Descargar el dataset COCO desde el Dashboard

  Como usuario del portal de anotación
  quiero exportar el dataset desde el Dashboard
  para obtener el archivo COCO sin realizar pasos manuales.

  Scenario: Mostrar la acción de exportación COCO
    Given que el usuario está en el Dashboard
    When visualiza las acciones principales
    Then debe existir una acción llamada "Exportar COCO"

  Scenario: La acción utiliza el endpoint de exportación
    Given que el usuario está en el Dashboard
    When selecciona la acción "Exportar COCO"
    Then la descarga debe dirigirse al endpoint "/export/coco"