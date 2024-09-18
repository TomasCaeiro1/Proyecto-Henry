import pandas as pd
import calendar
from fastapi import FastAPI, HTTPException
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()


df = pd.read_csv('movies_dataset_modificado.csv')

# Mapeo de nombres de meses en inglés a español
meses_ingles_a_espanol = {
    'January': 'enero', 'February': 'febrero', 'March': 'marzo', 'April': 'abril',
    'May': 'mayo', 'June': 'junio', 'July': 'julio', 'August': 'agosto',
    'September': 'septiembre', 'October': 'octubre', 'November': 'noviembre', 'December': 'diciembre'
}

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

def cantidad_filmaciones_mes(mes):
    df_filtrado = df[df['release_date'].notna()].copy()
    df_filtrado['month_name'] = df_filtrado['release_date'].dt.month.apply(lambda x: calendar.month_name[x])
    df_filtrado['mes_espanol'] = df_filtrado['month_name'].map(meses_ingles_a_espanol)
    cantidad = df_filtrado[df_filtrado['mes_espanol'] == mes].shape[0]
    return cantidad

@app.get("/cantidad_peliculas_mes/{mes}")
def get_cantidad_peliculas(mes: str):
    mes = mes.lower()
    if mes not in meses_ingles_a_espanol.values():
        raise HTTPException(status_code=400, detail="Mes inválido")
    
    cantidad = cantidad_filmaciones_mes(mes)
    return {"mes": mes, "cantidad_peliculas": cantidad} 


df = pd.read_csv('movies_dataset_modificado.csv')

# Mapeo de los días de la semana en inglés a español
dias_ingles_a_espanol = {
    'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miercoles', 
    'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sabado', 'Sunday': 'domingo'
}

# Convertir la columna 'release_date' a formato datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

def cantidad_filmaciones_dia(dia):
    # Filtrar filas donde 'release_date' es válido (no NaT)
    df_filtrado = df[df['release_date'].notna()].copy()
    
    # Extraer el nombre del día en inglés
    df_filtrado['day_name'] = df_filtrado['release_date'].dt.day_name()

    # Mapear el nombre del día de inglés a español
    df_filtrado['dia_espanol'] = df_filtrado['day_name'].map(dias_ingles_a_espanol)
    
    # Contar cuántas películas fueron estrenadas en el día dado en español
    cantidad = df_filtrado[df_filtrado['dia_espanol'] == dia].shape[0]
    
    return cantidad

@app.get("/cantidad_peliculas_dia/{dia}")
def get_cantidad_peliculas(dia: str):
    dia = dia.lower()
    if dia not in dias_ingles_a_espanol.values():
        raise HTTPException(status_code=400, detail="Día inválido")
    
    cantidad = cantidad_filmaciones_dia(dia)
    return {"dia": dia, "cantidad_peliculas": cantidad}


df = pd.read_csv('movies_dataset_modificado.csv')

# Convertir la columna 'release_date' a formato datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

def score_titulo(titulo_de_la_filmacion):
    # Buscar la película por el título ingresado
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]

    if pelicula.empty:
        return "No se encontró la película con ese título."
    
    # Obtener el título, año y popularidad (score)
    titulo = pelicula['title'].values[0]
    year = pelicula['release_date'].dt.year.values[0] if pd.notna(pelicula['release_date'].values[0]) else "Año desconocido"
    score = pelicula['popularity'].values[0] if 'popularity' in pelicula.columns else "Score no disponible"
    
    return {"titulo": titulo, "año": year, "score": score}

@app.get("/score_pelicula/{titulo}")
def get_score_pelicula(titulo: str):
    resultado = score_titulo(titulo)
    
    if isinstance(resultado, str):
        raise HTTPException(status_code=404, detail=resultado)
    
    return resultado



df = pd.read_csv('movies_dataset_modificado.csv')

def votos_titulo(titulo_de_la_filmacion):
    # Buscar la película por el título ingresado
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    if pelicula.empty:
        return "No se encontró la película con ese título."
    
    # Asegurarse de que las columnas necesarias están presentes
    if 'vote_count' not in pelicula.columns or 'vote_average' not in pelicula.columns:
        return "Datos de votación no disponibles."
    
    # Verificar que tenga al menos 2000 valoraciones
    if pelicula['vote_count'].values[0] < 2000:
        return f"La película {pelicula['title'].values[0]} no cumple con el mínimo de 2000 valoraciones."
    
    # Obtener el título, año, cantidad de votos y promedio de votaciones
    titulo = pelicula['title'].values[0]
    year = pelicula['release_date'].dt.year.values[0] if pd.notna(pelicula['release_date'].values[0]) else "Año desconocido"
    votos = pelicula['vote_count'].values[0]
    promedio_votos = pelicula['vote_average'].values[0]
    
    return {"titulo": titulo, "año": year, "cantidad_votos": votos, "promedio_votos": promedio_votos}

@app.get("/votos_pelicula/{titulo}")
def get_votos_pelicula(titulo: str):
    resultado = votos_titulo(titulo)
    
    if isinstance(resultado, str):
        raise HTTPException(status_code=404, detail=resultado)
    
    return resultado


movies_df = pd.read_csv('movies_dataset_modificado.csv')
credits_df = pd.read_csv('credits.csv')

# Asegurarnos de que las columnas de budget y revenue existan para calcular el retorno
if 'revenue' in movies_df.columns and 'budget' in movies_df.columns:
    # Convertir las columnas 'revenue' y 'budget' a valores numéricos
    movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')
    movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
    
    # Crear una columna "return" calculando la relación entre revenue y budget, evitando la división por cero
    movies_df['return'] = movies_df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)
else:
    raise ValueError("No se encontraron las columnas 'revenue' o 'budget' en el dataset de películas.")

# Asegurarnos de que las columnas 'id' en ambos datasets son del mismo tipo
movies_df['id'] = movies_df['id'].astype(str)
credits_df['id'] = credits_df['id'].astype(str)

# Función para obtener el éxito de un actor
def get_actor(nombre_actor):
    # Filtrar las filas donde el actor aparece en el cast 
    peliculas_actor = credits_df[credits_df['cast'].str.contains(nombre_actor, case=False, na=False)]
    
    if peliculas_actor.empty:
        return f"No se encontraron películas para el actor {nombre_actor}."
    
    # Unir las películas del actor con los detalles de las películas
    peliculas_actor = peliculas_actor.merge(movies_df[['id', 'title', 'return']], left_on='id', right_on='id')
    
    # Calcular la cantidad de películas
    cantidad_peliculas = peliculas_actor.shape[0]
    
    # Calcular el retorno total y el promedio de retorno
    retorno_total = peliculas_actor['return'].sum()
    promedio_retorno = peliculas_actor['return'].mean()
    
    return {"actor": nombre_actor, "cantidad_peliculas": cantidad_peliculas, "retorno_total": retorno_total, "promedio_retorno": promedio_retorno}

@app.get("/exito_actor/{nombre_actor}")
def obtener_exito_actor(nombre_actor: str):
    resultado = get_actor(nombre_actor)
    
    if isinstance(resultado, str):  # Si el resultado es un mensaje de error
        raise HTTPException(status_code=404, detail=resultado)
    
    return resultado



movies_df = pd.read_csv('movies_dataset_modificado.csv')
credits_df = pd.read_csv('credits.csv')

# Asegurarnos de que las columnas de budget y revenue existan para calcular el retorno
if 'revenue' in movies_df.columns and 'budget' in movies_df.columns:
    # Convertir las columnas 'revenue' y 'budget' a valores numéricos
    movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')
    movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
    
    # Crear una columna "return" calculando la relación entre revenue y budget, evitando la división por cero
    movies_df['return'] = movies_df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)
else:
    raise ValueError("No se encontraron las columnas 'revenue' o 'budget' en el dataset de películas.")

# Asegurarnos de que las columnas 'id' en ambos datasets son del mismo tipo
movies_df['id'] = movies_df['id'].astype(str)
credits_df['id'] = credits_df['id'].astype(str)

# Función para obtener los datos de un director
def get_director(nombre_director):
    # Convertir la columna 'crew' que está en formato string a listas de diccionarios
    credits_df['crew'] = credits_df['crew'].apply(ast.literal_eval)
    
    # Filtrar las películas donde el director está en el crew con el trabajo de "Director"
    peliculas_director = credits_df[credits_df['crew'].apply(lambda x: any(d['name'] == nombre_director and d['job'] == 'Director' for d in x))]
    
    if peliculas_director.empty:
        return f"No se encontraron películas para el director {nombre_director}."
    
    # Unir las películas del director con los detalles de las películas
    peliculas_director = peliculas_director.merge(movies_df[['id', 'title', 'release_date', 'budget', 'revenue', 'return']], left_on='id', right_on='id')
    
    # Lista para almacenar los resultados
    detalles_peliculas = []
    
    # Iterar por cada película del director
    for _, row in peliculas_director.iterrows():
        titulo = row['title']
        fecha_lanzamiento = row['release_date']
        retorno_individual = row['return']
        costo = row['budget']
        ganancia = row['revenue'] - row['budget']  # Ganancia = revenue - budget
        
        detalles_peliculas.append(f"Película: {titulo}, Fecha de lanzamiento: {fecha_lanzamiento}, Retorno: {retorno_individual:.2f}, Costo: {costo}, Ganancia: {ganancia}")
    
    # Calcular el retorno total del director
    retorno_total = peliculas_director['return'].sum()
    
    resultado = f"El director {nombre_director} ha dirigido {len(peliculas_director)} películas, con un retorno total de {retorno_total:.2f}.\n"
    resultado += "\n".join(detalles_peliculas)
    
    return resultado

@app.get("/directores/{nombre_director}")
def obtener_director(nombre_director: str):
    resultado = get_director(nombre_director)
    
    if isinstance(resultado, str):  # Si el resultado es un mensaje de error
        raise HTTPException(status_code=404, detail=resultado)
    
    return {"resultado": resultado}



df = pd.read_csv('movies_dataset_modificado.csv')

# Asegúrate de que los títulos de las películas no tengan valores nulos
df = df[df['title'].notna()]

# Función de recomendación basada en similitud
def recomendacion(titulo):
    # Asegurarse de que el título esté en minúsculas
    titulo = titulo.lower()
    
    # Verificar si la película existe en el dataset
    if titulo not in df['title'].str.lower().values:
        return "Película no encontrada."
    
    # Usar CountVectorizer para vectorizar los títulos de las películas
    vectorizer = CountVectorizer().fit_transform(df['title'].str.lower())
    vectors = vectorizer.toarray()
    
    # Calcular la similitud del coseno entre las películas
    cosine_sim = cosine_similarity(vectors)
    
    # Obtener el índice de la película dada
    idx = df[df['title'].str.lower() == titulo].index[0]
    
    # Obtener la lista de similitudes para esa película
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenar las películas por la puntuación de similitud (de mayor a menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtener los índices de las 5 películas más similares
    sim_indices = [i[0] for i in sim_scores[1:6]]
    
    # Devolver los títulos de las películas recomendadas
    recommended_movies = df['title'].iloc[sim_indices].values.tolist()
    
    return recommended_movies

# Crear el endpoint en FastAPI
@app.get("/recomendacion/{titulo}")
def obtener_recomendaciones(titulo: str):
    return recomendacion(titulo)
