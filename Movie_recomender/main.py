import streamlit as st
import numpy as np
import pandas as pd
import difflib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('processed.csv')

def calculate_similarity(content):
    vectorizer = TfidfVectorizer(lowercase=True,stop_words='english')
    tdif_matrix = vectorizer.fit_transform(content)
    similarity = cosine_similarity(tdif_matrix)
    return similarity

def get_poster(id):
    API_KEY = 'get your api key from tmdb'

    url = f"https://api.themoviedb.org/3/movie/{id}?api_key={API_KEY}"
    response = requests.get(url).json()

    poster_path = response.get('poster_path')
    if poster_path:
        image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        return image_url

def recomend(title, similarity, n_tops=6):
    list_of_all_titles = df['title'].tolist()
    find_close_match = difflib.get_close_matches(title, list_of_all_titles)

    if not find_close_match:
        return pd.DataFrame(columns=['title', 'score', 'id'])

    close_match = find_close_match[0]
    idx = df[df.title == close_match].index.values[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = df.iloc[movie_indices]

    return similar_movies[['title', 'score', 'id']].head(n_tops)

# similarity scores
similarity = calculate_similarity(df['combined_content'])
# input
movie_input = st.text_input(label='',placeholder='type a movie...')

if movie_input:
    get_movie_data = recomend(movie_input, similarity)
    if get_movie_data.empty:
        st.warning("No similar movies found.")
    else:
        try:
            num_movies = len(get_movie_data)
            num_columns = 3
            rows = (num_movies + num_columns - 1) // num_columns 
            for row in range(rows):
                cols = st.columns(num_columns)
                for col in range(num_columns):
                    idx = row * num_columns + col
                    if idx < num_movies:
                        with cols[col]:
                            with st.container(border=True):
                                title = get_movie_data['title'].values[idx]
                                score = get_movie_data['score'].values[idx]
                                st.markdown(f"**{title}** &nbsp;&nbsp; _({score:.2f})_", unsafe_allow_html=True)
                                st.image(get_poster(get_movie_data['id'].values[idx]))
        except Exception as e:
            st.error("Something went wrong.")

