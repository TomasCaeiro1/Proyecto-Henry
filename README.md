# Proyecto Individual Henry
Este proyecto es una API que utiliza FastAPI para ofrecer diversos endpoints que permiten consultar información sobre películas, incluidas recomendaciones basadas en similitud, cantidad de películas por mes y día, y más. Está diseñado para trabajar con un conjunto de datos modificado de películas.

## Funcionalidades
- Cantidad de Películas por Mes: Consulta la cantidad de películas lanzadas en un mes específico.
- Cantidad de Películas por Día: Obtén cuántas películas se lanzaron en un día de la semana específico.
- Éxito de un Actor: Proporciona el éxito de un actor, incluyendo el retorno de sus películas.
- Votos de una Película: Devuelve la cantidad de votos y el promedio de calificación de una película.
- Éxito de una Película: Devuelve información sobre el éxito de una película, basado en métricas como el retorno financiero, la cantidad de votos y el promedio de calificación.
- Éxito de un Director: Devuelve el éxito de un director basándose en su filmografía.
- Recomendaciones de Películas: A partir de un título de película, el sistema sugiere películas similares utilizando la similitud del coseno.
  
## Tecnologias utilizadas
- FastAPI: Framework para construir la API.
- Pandas: Para manipulación de datos.
- Scikit-learn: Para cálculos de similitud de películas.
- CountVectorizer: Utilizado para vectorizar los títulos de las películas.
- Cosine Similarity: Métrica utilizada para recomendar películas similares.
