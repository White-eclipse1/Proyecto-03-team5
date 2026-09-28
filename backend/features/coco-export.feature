@SPEC-COCO-001
Feature: Exportaci├│n del dataset en formato COCO

  Como responsable del dataset
  quiero exportar im├ígenes, anotaciones y categor├¡as en formato COCO
  para alimentar el pipeline de entrenamiento sin transformaciones manuales.

  Scenario: El JSON tiene las tres secciones obligatorias
    Given un dataset con im├ígenes, anotaciones y categor├¡as
    When se construye el documento COCO
    Then el resultado tiene las claves "images", "annotations" y "categories"

  Scenario: Cada anotaci├│n usa el bbox en orden [x, y, width, height]
    Given una anotaci├│n con bboxX=10, bboxY=20, bboxWidth=100, bboxHeight=50
    When se construye el documento COCO
    Then su bbox es [10, 20, 100, 50]

  Scenario: El ├írea de cada anotaci├│n es coherente con el bbox
    Given una anotaci├│n con bboxWidth=100 y bboxHeight=50
    When se construye el documento COCO
    Then su area es 5000
    And su area es igual a bbox[2] por bbox[3]

  Scenario: El campo iscrowd siempre est├í presente como 0 o 1
    Given una anotaci├│n con isCrowd=false
    When se construye el documento COCO
    Then su iscrowd es 0

  Scenario: Los IDs son consistentes entre secciones
    Given un dataset con im├ígenes y categor├¡as
    When se construye el documento COCO
    Then cada annotation.image_id existe en la secci├│n images
    And cada annotation.category_id existe en la secci├│n categories

  Scenario: Las im├ígenes sin anotaciones tambi├®n se incluyen
    Given una imagen sin ninguna anotaci├│n
    When se construye el documento COCO
    Then la imagen aparece en la secci├│n images
