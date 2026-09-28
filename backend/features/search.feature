@SPEC-SEARCH-001
Feature: B├║squeda de im├ígenes por clases con operadores

  Como anotador
  quiero buscar im├ígenes combinando clases con AND y OR
  para localizar r├ípidamente las escenas que me interesan.

  Scenario: Buscar una sola clase
    When se parsea la consulta "car"
    Then los t├®rminos son ["car"]
    And el operador es "AND"

  Scenario: Buscar dos clases con AND
    When se parsea la consulta "car AND person"
    Then los t├®rminos son ["car", "person"]
    And el operador es "AND"

  Scenario: Buscar dos clases con OR
    When se parsea la consulta "car OR person"
    Then los t├®rminos son ["car", "person"]
    And el operador es "OR"

  Scenario: Ignorar espacios sobrantes
    When se parsea la consulta "   car    AND   person  "
    Then los t├®rminos son ["car", "person"]

  Scenario: Normalizar t├®rminos a min├║sculas
    When se parsea la consulta "Car AND PERSON"
    Then los t├®rminos son ["car", "person"]

  Scenario: Una consulta vac├¡a no aplica b├║squeda
    When se parsea la consulta "   "
    Then el resultado es nulo

  Scenario: No se permite mezclar operadores distintos
    When se parsea la consulta "car AND person OR dog"
    Then el parseo falla con un error

  Scenario: No se permite un operador sin t├®rmino a la derecha
    When se parsea la consulta "car AND"
    Then el parseo falla con un error

  Scenario: No se permite un operador sin t├®rmino a la izquierda
    When se parsea la consulta "AND car"
    Then el parseo falla con un error

  Scenario: Una clase que contiene la palabra AND no se parte
    When se parsea la consulta "android"
    Then los t├®rminos son ["android"]
