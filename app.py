import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv('movies.csv')

# Select useful columns
movies = movies[['movieId', 'title', 'genres']]

# Remove missing values
movies.dropna(inplace=True)

# Convert genres into vectors
cv = CountVectorizer(stop_words='english')

matrix = cv.fit_transform(movies['genres'])

# Calculate similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie_name):

    # Check movie exists
    if movie_name not in movies['title'].values:
        print("Movie not found!")
        return

    # Get movie index
    movie_index = movies[movies['title'] == movie_name].index[0]

    # Get similarity scores
    distances = similarity[movie_index]

    # Sort movies
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nRecommended movies for {movie_name}:\n")

    # Print recommendations
    for movie in movies_list:
        print(movies.iloc[movie[0]].title)

# Show sample movies
print("Some available movies:\n")
print(movies['title'].head(20))

# User input
movie_name = input("\nEnter movie name: ")

# Call recommendation function
recommend(movie_name)