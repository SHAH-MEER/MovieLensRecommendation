import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def build_advanced_recommender(ratings_df, n_components=100):
    """Build a hybrid recommendation system using TruncatedSVD and content features"""
    # Create user-movie matrix
    user_movie_matrix = ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    # Apply TruncatedSVD
    svd = TruncatedSVD(n_components=min(n_components, min(user_movie_matrix.shape)-1))
    movie_features = svd.fit_transform(user_movie_matrix.T)
    
    # Compute movie similarity matrix
    movie_similarity = cosine_similarity(movie_features)
    
    return pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

def get_recommendations(movies_df, similarity_matrix, selected_movie_ids, n_recommendations=10):
    """Get movie recommendations using the similarity matrix"""
    # Get similarity scores for selected movies
    similar_scores = similarity_matrix[selected_movie_ids].mean(axis=1)
    similar_scores = pd.Series(similar_scores, index=similarity_matrix.index)
    
    # Remove selected movies
    similar_scores = similar_scores.drop(selected_movie_ids)
    
    # Get top recommendations
    top_movie_ids = similar_scores.nlargest(n_recommendations).index
    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)].copy()
    
    # Add similarity scores
    recommendations['similarity_score'] = recommendations['movieId'].map(similar_scores)
    
    return recommendations.sort_values('similarity_score', ascending=False)

def get_genre_preferences(selected_movies, movies_df):
    """Extract genre preferences from selected movies"""
    genres = []
    for _, movie in movies_df[movies_df['title'].isin(selected_movies)].iterrows():
        genres.extend(movie['genres'].split('|'))
    return pd.Series(genres).value_counts()

def get_cold_start_recommendations(movies_df, ratings_df, genre_preferences=None, n_recommendations=10):
    """Get recommendations for new users based on popularity and genres"""
    # Calculate popularity scores
    popularity = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    popularity.columns = ['movieId', 'rating_count', 'rating_mean']
    popularity['popularity_score'] = popularity['rating_count'] * popularity['rating_mean']
    
    # Merge with movie information
    recommendations = movies_df.merge(popularity, on='movieId')
    
    if genre_preferences is not None and not genre_preferences.empty:
        # Weight by genre preference
        recommendations['genre_score'] = recommendations['genres'].apply(
            lambda x: sum(genre_preferences.get(genre, 0) for genre in x.split('|'))
        )
        recommendations['final_score'] = recommendations['popularity_score'] * (1 + recommendations['genre_score'])
        return recommendations.nlargest(n_recommendations, 'final_score')
    
    return recommendations.nlargest(n_recommendations, 'popularity_score')

def get_recommendation_explanation(movie, similarity_score, movies_df, original_movies):
    """Generate explanation for why a movie was recommended"""
    explanation = []
    
    # Genre similarity
    movie_genres = set(movie['genres'].split('|'))
    original_genres = set()
    for title in original_movies:
        original_genres.update(
            movies_df[movies_df['title'] == title].iloc[0]['genres'].split('|')
        )
    
    common_genres = movie_genres.intersection(original_genres)
    if common_genres:
        explanation.append(f"Shares genres: {', '.join(common_genres)}")
    
    # Similarity score explanation
    if similarity_score >= 0.8:
        explanation.append("Very strong match")
    elif similarity_score >= 0.6:
        explanation.append("Strong match")
    elif similarity_score >= 0.4:
        explanation.append("Moderate match")
    else:
        explanation.append("Weak match")
    
    return " | ".join(explanation)
