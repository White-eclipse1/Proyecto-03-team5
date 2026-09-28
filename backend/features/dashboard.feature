@SPEC-DASH-001
Feature: M├®tricas del dashboard calculadas desde la base de datos

  Como responsable del dataset
  quiero ver totales y progreso reales
  para saber cu├ínto falta por anotar sin revisar la base de datos a mano.

  Scenario: El total de im├ígenes suma todos los estados
    Given 2 im├ígenes pendientes, 1 en progreso y 3 completadas
    When se calcula el resumen del dashboard
    Then imagesUploaded es 6

  Scenario: Solo las completadas cuentan como anotadas
    Given 2 im├ígenes pendientes, 1 en progreso y 3 completadas
    When se calcula el resumen del dashboard
    Then imagesAnnotated es 3

  Scenario: El progreso separa lo terminado de lo pendiente
    Given 2 im├ígenes pendientes, 1 en progreso y 3 completadas
    When se calcula el resumen del dashboard
    Then annotationProgress.annotated es 3
    And annotationProgress.pending es 3

  Scenario: El progreso siempre cuadra con el total
    Given cualquier distribuci├│n de estados
    When se calcula el resumen del dashboard
    Then annotated m├ís pending es igual a imagesUploaded

  Scenario: Los objetos por clase conservan nombre y color
    Given una categor├¡a "car" con color "#3498DB" y 4 anotaciones
    When se calcula el resumen del dashboard
    Then objectsPerClass incluye car con color #3498DB y count 4

  Scenario: Las miniaturas se sirven por el backend
    Given una imagen con id 7
    When se calcula el resumen del dashboard
    Then su thumbnailUrl es "/images/7/file"

  Scenario: Un dataset vac├¡o devuelve ceros y no falla
    Given ninguna imagen registrada
    When se calcula el resumen del dashboard
    Then todos los totales son 0
    And objectsPerClass est├í vac├¡o
