import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Load datasets
movies = pd.read_csv() #CSV file path for movies dataset, add file name
ratings = pd.read_csv() #CSV file path for ratings dataset, add the file name

# Limit dataset size for better performance
ratings = ratings.head(100000)  # Use first 100,000 ratings
movies = movies[movies['movieId'].isin(ratings['movieId'])]  # Keep only rated movies

# Data preprocessing
movie_ratings = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)
sparse_user_movie = csr_matrix(user_movie_matrix.values)
movies['genres'] = movies['genres'].str.replace('|', ' ')

def collaborative_filtering_recommendations(title, n_recommendations=5):
    if title not in user_movie_matrix.columns:
        return f"Movie '{title}' not found in database."
    
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(sparse_user_movie.T)
    
    movie_index = list(user_movie_matrix.columns).index(title)
    distances, indices = model_knn.kneighbors(
        user_movie_matrix.iloc[:, movie_index].values.reshape(1, -1), 
        n_neighbors=n_recommendations+1
    )
    
    return [user_movie_matrix.columns[indices.flatten()[i]] for i in range(1, len(indices.flatten()))]

def content_based_recommendations(title, n_recommendations=5):
    if title not in movies['title'].values:
        return f"Movie '{title}' not found in database."
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
    
    return movies['title'].iloc[movie_indices].tolist()

def main():
    print("Movie Recommendation System") #Tittle
    print("="*30)
    
    while True:
        print("\nOptions:")
        print("1. Get collaborative filtering recommendations") #Ask user for collaborative filtering recommendations
        print("2. Get content-based recommendations") #Ask user for content-based recommendations
        print("3. Exit") # for exiting from the options
        #User choice
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            movie_title = input("Enter movie title: ")
            print("\nCollaborative Filtering Recommendations:")
            print(collaborative_filtering_recommendations(movie_title))
            
        elif choice == '2':
            movie_title = input("Enter movie title: ")
            print("\nContent-Based Recommendations:")
            print(content_based_recommendations(movie_title))
                
        elif choice == '3':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
