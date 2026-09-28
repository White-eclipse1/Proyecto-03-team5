@SPEC-ANNOT-001
Feature: Validaci├│n de bounding boxes

  Como usuario del portal de anotaci├│n
  quiero que las cajas sean validadas antes de guardarse
  para que el dataset exportado sea geom├®tricamente consistente.

  Scenario: Aceptar una bounding box v├ílida dentro del canvas
    Given una imagen de 640x480 p├¡xeles
    And una categor├¡a existente con id 1
    When se env├¡a una caja con bboxX=10, bboxY=20, bboxWidth=100, bboxHeight=50
    Then la validaci├│n debe aceptar la caja
    And el ├írea calculada debe ser 5000

  Scenario: Rechazar una caja con origen negativo
    Given una imagen de 640x480 p├¡xeles
    When se env├¡a una caja con bboxX=-5, bboxY=0, bboxWidth=100, bboxHeight=50
    Then la validaci├│n debe rechazar la caja

  Scenario: Rechazar una caja con dimensiones cero o negativas
    Given una imagen de 640x480 p├¡xeles
    When se env├¡a una caja con bboxX=0, bboxY=0, bboxWidth=0, bboxHeight=50
    Then la validaci├│n debe rechazar la caja

  Scenario: Rechazar una caja que excede el ancho de la imagen
    Given una imagen de 640x480 p├¡xeles
    When se env├¡a una caja con bboxX=600, bboxY=0, bboxWidth=100, bboxHeight=50
    Then la validaci├│n debe rechazar la caja

  Scenario: Rechazar una caja que excede el alto de la imagen
    Given una imagen de 640x480 p├¡xeles
    When se env├¡a una caja con bboxX=0, bboxY=450, bboxWidth=100, bboxHeight=50
    Then la validaci├│n debe rechazar la caja

  Scenario: Rechazar una caja sin categor├¡a v├ílida (categoryId <= 0)
    Given una imagen de 640x480 p├¡xeles
    When se env├¡a una caja con categoryId=0
    Then la validaci├│n debe rechazar la caja

  Scenario: Mover una caja sin categoryId en el patch no debe cambiar su categoría
    Given una anotación existente con categoryId=2
    When se envía un patch que solo trae bboxX=30
    Then la anotación resultante conserva categoryId=2

  Scenario: Reclasificar una caja sin bbox en el patch no debe mover su geometría
    Given una anotación existente con bboxX=10, bboxY=20, bboxWidth=100, bboxHeight=50
    When se envía un patch que solo trae categoryId=9
    Then la anotación resultante conserva bboxX=10, bboxY=20, bboxWidth=100, bboxHeight=50
