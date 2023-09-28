import nltk
import unicodedata
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from utils.string import name_treat, gruped_metadados, tagline_treat, remover_pontuacao
import pandas as pd
nltk.download("stopwords")

def data():
    movies = pd.read_csv('https://raw.githubusercontent.com/alexvaroz/data_science_alem_do_basico/master/tmdb_movies_data.csv')
#    movies = movies[movies.vote_count >= 50]
    movies = movies[movies['vote_count'] >= 50]
    movies.reset_index(drop=True, inplace=True)
    movies.original_title = movies.original_title.apply(lambda x: unicodedata.normalize('NFKD', x))
    return movies

def recommendation_by_content(title, cosine_sim, df, indice):

    # Obtenção do índice pelo título
    idx = indice[title]

    # recuperação dos valores filtrados do índice pela matriz de similaridade
    # É passada uma lista
    sim_scores = enumerate(list(cosine_sim[idx]))

    # A lista é ordenada
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # São separados os 10 maiores valores, excluindo o primeiro

    sim_scores = sim_scores[1:11]
    # São recuperados os índices relativos aos 10 filmes

    movie_indices = [i[0] for i in sim_scores]
    # COm os índices é possível obter as informações sobre os filmes
    return list(df['original_title'].loc[movie_indices])

def simple_movie_recommendation(title):
    movies = data()

    # Substituição dos valores NAN
    movies["overview"] = movies["overview"].fillna('')

    # Instancia um objeto TF-IDF Vectorizer removendo as stopwords em inglês. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')

    # Cria a matriz TF-IDF matriz executando o método "fit_transform"
    tfidf_matriz = tfidf.fit_transform(movies['overview'])

    # Obter a matriz com a similaridade por cosseno entre os registros
    cosine_sim = linear_kernel(tfidf_matriz)
    reversed_index = pd.Series(movies.index, index=movies.original_title).drop_duplicates()

    return recommendation_by_content(title, cosine_sim, movies, reversed_index)


def aggregated_movie_recomendation(title):
    movies = data()

    # Generos
    movies['genres'] = movies['genres'].fillna('')
    movies['genres_lst'] = movies.apply(lambda row: row['genres'].split('|') if row['genres'] != '' else [], axis=1)

    # Diretor
    movies['director'] = movies['director'].fillna('')
    movies['director_lst'] = movies.apply(lambda row: row['director'].split('|') if row['director'] != '' else [], axis=1)
    movies['director_lst'] = movies.apply(lambda row: name_treat(row['director_lst']), axis=1)

    # Elenco
    movies['cast'] = movies['cast'].fillna('')
    movies['cast_lst'] = movies.apply(lambda row: row['cast'].split('|') if row['cast'] != '' else [], axis=1)
    movies['cast_lst'] = movies['cast_lst'].apply(lambda x: x[:3])
    movies['cast_lst'] = movies.apply(lambda row: name_treat(row['cast_lst']), axis=1)

    # Palavras chaves
    movies['keywords'] = movies['keywords'].fillna('')
    movies['keywords_lst'] = movies.apply(lambda row: row['keywords'].split('|') if row['keywords'] != '' else [], axis=1)
    movies['keywords_lst'] = movies['keywords_lst'].apply(lambda x: x[:3])
    movies['keywords_lst'] = movies.apply(lambda row: name_treat(row['keywords_lst']), axis=1)

    # Taglines
    movies['tagline_str'] = movies.apply(lambda row: tagline_treat(row['tagline']), axis=1)

    # Consolidação
    movies['metadados_grouping'] = movies.apply(gruped_metadados, axis=1)

    # Cria a matriz TF-IDF matriz executando o método "fit_transform"
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matriz_metadados = tfidf.fit_transform(movies['metadados_grouping'])

    # Obter a matriz com a similaridade por cosseno entre os registros
    cosine_sim_metadados = linear_kernel(tfidf_matriz_metadados)
    reversed_index = pd.Series(movies.index, index=movies.original_title).drop_duplicates()

    return recommendation_by_content(title, cosine_sim_metadados, movies, reversed_index)

