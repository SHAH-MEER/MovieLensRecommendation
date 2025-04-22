import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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
