@SPEC-UPLOAD-001
Feature: Carga de imágenes

  Como usuario del portal de anotación
  quiero subir imágenes válidas
  para poder anotarlas posteriormente.

  Scenario: Aceptar una imagen JPEG válida
    Given una imagen con tipo "image/jpeg"
    And su tamaño está dentro del límite permitido
    When se valida la imagen para su carga
    Then la imagen debe ser aceptada

  Scenario: Rechazar un archivo que no sea imagen
    Given un archivo con tipo "application/pdf"
    When se valida el archivo para su carga
    Then la carga debe ser rechazada

  Scenario: Rechazar una imagen demasiado grande
    Given una imagen con un tamaño mayor al límite permitido
    When se valida la imagen para su carga
    Then la carga debe ser rechazada