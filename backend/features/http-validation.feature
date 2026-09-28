@SPEC-VALID-001
Feature: Validación de la frontera HTTP con Zod

  Como responsable del backend
  quiero que todo dato externo se valide con Zod antes de llegar a la base de datos
  para que un dato mal formado devuelva 400 con un mensaje claro
  en lugar de propagarse hasta la capa de datos.

  Scenario: Rechazar un route param decimal
    Given el route param "1.5"
    When se valida con el esquema de id
    Then la validación falla

  Scenario: Rechazar un route param que no es entero positivo
    Given el route param "abc"
    When se valida con el esquema de id
    Then la validación falla

  Scenario: Rechazar el route param cero
    Given el route param "0"
    When se valida con el esquema de id
    Then la validación falla

  Scenario: Rechazar un route param negativo
    Given el route param "-5"
    When se valida con el esquema de id
    Then la validación falla

  Scenario: Rechazar una cadena vacía como route param
    Given el route param ""
    When se valida con el esquema de id
    Then la validación falla

  Scenario: Aceptar y convertir un route param numérico
    Given el route param "7"
    When se valida con el esquema de id
    Then la validación es exitosa
    And el valor convertido es el número 7
