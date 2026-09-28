# SPEC-SEARCH-001 ÔÇö B├║squeda por clases con operadores

## Objetivo

Permitir buscar im├ígenes por las clases (categor├¡as) que contienen sus
anotaciones, usando operadores l├│gicos en la consulta, y resolver el
filtrado completamente en SQL sin traer filas intermedias a memoria.

## Sintaxis de la consulta

El par├ímetro `search` acepta nombres de clase separados por un operador
l├│gico en may├║sculas:

| Consulta            | Significado                                                  |
|---------------------|--------------------------------------------------------------|
| `car`               | im├ígenes con al menos una anotaci├│n de clase `car`           |
| `car AND person`    | im├ígenes que contienen **ambas** clases                      |
| `car OR person`     | im├ígenes que contienen **al menos una** de las dos           |
| `car AND person AND dog` | im├ígenes que contienen las tres clases                  |

## Reglas de parseo

1. Los operadores `AND` y `OR` se reconocen ├║nicamente en may├║sculas, como
   palabra completa, para no confundirlos con nombres de clase que los
   contengan (por ejemplo `android` no contiene el operador `AND`).
2. Los espacios sobrantes se descartan; `  car   AND  person ` es v├ílido.
3. Una consulta con un solo t├®rmino es v├ílida y no requiere operador.
4. Una consulta vac├¡a o con solo espacios se trata como "sin b├║squeda"
   (devuelve `null`, no error).
5. No se permite mezclar operadores distintos en la misma consulta
   (`car AND person OR dog` es un error): la precedencia ser├¡a ambigua.
6. No se permite un operador sin t├®rminos a ambos lados
   (`car AND`, `AND car`, `car AND AND person` son errores).
7. Los nombres de clase se normalizan a min├║sculas. La comparaci├│n en
   MariaDB es insensible a may├║sculas por la colaci├│n por defecto.

## Resoluci├│n en SQL

La b├║squeda **no** carga IDs ni filas en memoria para filtrarlas despu├®s.
Se traduce a subconsultas correlacionadas `EXISTS` sobre la tabla de
anotaciones unida a categor├¡as:

- Operador `AND`: una subconsulta `EXISTS` por cada t├®rmino, todas
  combinadas con `AND`. Esto exige que la imagen tenga al menos una
  anotaci├│n de cada clase.
- Operador `OR`: una ├║nica subconsulta `EXISTS` cuyo filtro usa
  `category.name IN (...)` con todos los t├®rminos.

Los ├¡ndices `annotations_image_id_idx` y `annotations_image_category_idx`
sirven a estas subconsultas.

## Combinaci├│n con otros filtros

`search` es combinable con `status`, `categoryId`, `filename`, `dateFrom`,
`dateTo` y la paginaci├│n. Todos los filtros se aplican en el mismo
`WHERE`, y el conteo total para la paginaci├│n usa las mismas condiciones.

## Flujo esperado

UI (GET /images?search=...) ÔåÆ Logic (parseSearchQuery + Zod) ÔåÆ Data (EXISTS en SQL)
