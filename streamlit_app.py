import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page settings
st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 AI Movie Recommendation System")

st.write("Discover movies using Machine Learning")

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Select useful columns
movies = movies[['movieId', 'title', 'genres']]

# Remove missing values
movies.dropna(inplace=True)

# -----------------------------
# TOP RATED MOVIES
# -----------------------------

st.subheader("⭐ Top Rated Movies")

average_ratings = ratings.groupby(
    'movieId'
)['rating'].mean()

rating_count = ratings.groupby(
    'movieId'
)['rating'].count()

rating_data = pd.DataFrame({
    'average_rating': average_ratings,
    'rating_count': rating_count
})

top_movies = movies.merge(
    rating_data,
    on='movieId'
)

top_movies = top_movies[
    top_movies['rating_count'] > 50
]

top_movies = top_movies.sort_values(
    by='average_rating',
    ascending=False
)

for movie in top_movies.head(10)['title'].values:
    st.write("🎥", movie)

st.markdown("---")

# -----------------------------
# GENRE FILTER
# -----------------------------

all_genres = []

for genre in movies['genres']:

    genres_split = genre.split('|')

    for g in genres_split:
        all_genres.append(g)

genre_list = sorted(list(set(all_genres)))

selected_genre = st.selectbox(
    "🎭 Select Genre",
    genre_list
)

filtered_movies = movies[
    movies['genres'].str.contains(selected_genre)
]

# -----------------------------
# VECTORIZATION
# -----------------------------

cv = CountVectorizer(stop_words='english')

matrix = cv.fit_transform(
    filtered_movies['genres']
)

similarity = cosine_similarity(matrix)

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------

def recommend(movie_name):

    movie_index = filtered_movies[
        filtered_movies['title'] == movie_name
    ].index[0]

    movie_position = list(
        filtered_movies.index
    ).index(movie_index)

    distances = similarity[movie_position]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for movie in movies_list:

        recommended_movies.append(
            filtered_movies.iloc[movie[0]].title
        )

    return recommended_movies

# -----------------------------
# SEARCH BOX
# -----------------------------

movie_search = st.text_input(
    "🔍 Search Movie"
)

# Matching movies
matching_movies = filtered_movies[
    filtered_movies['title']
    .str.contains(movie_search,
                  case=False,
                  na=False)
]

# Dropdown
selected_movie = st.selectbox(
    "🎬 Select Movie",
    matching_movies['title'].values
)

# Recommend button
if st.button("🚀 Recommend"):

    recommendations = recommend(
        selected_movie
    )

    st.subheader(
        "🎯 Recommended Movies"
    )

    for i, movie in enumerate(
        recommendations,
        start=1
    ):

        st.write(f"{i}. {movie}")

# Footer
st.markdown("---")

st.caption(
    "Built with Python, Scikit-learn & Streamlit"
)