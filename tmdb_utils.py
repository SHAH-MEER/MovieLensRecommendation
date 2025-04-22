import streamlit as st
import requests
import pandas as pd
from collections import defaultdict

# Use secret from Streamlit instead of hardcoded key
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_credits(movie_id):
    """Fetch cast and crew data for a movie"""
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    response = requests.get(url, params={"api_key": TMDB_API_KEY})
    return response.json() if response.ok else None

def get_movie_details(movie_id):
    """Fetch movie details including revenue and budget"""
    url = f"{BASE_URL}/movie/{movie_id}"
    response = requests.get(url, params={"api_key": TMDB_API_KEY})
    return response.json() if response.ok else None

def get_movie_poster_url(movie_details):
    """Get movie poster URL from TMDB"""
    if movie_details and movie_details.get('poster_path'):
        return f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
    return None

def build_collaboration_network(movies_df, links_df, max_movies=30):
    """Build actor and director collaboration network with improved filtering"""
    actor_collaborations = defaultdict(lambda: defaultdict(int))
    director_actors = defaultdict(lambda: defaultdict(int))
    
    # Get more recent movies first by sorting on movieId (assuming higher IDs are newer)
    sample_movies = links_df.merge(movies_df, on='movieId').sort_values('movieId', ascending=False).head(max_movies)
    
    with st.spinner("Analyzing movie collaborations..."):
        for _, movie in sample_movies.iterrows():
            credits = get_movie_credits(movie['tmdbId'])
            
            if not credits or 'cast' not in credits or 'crew' not in credits:
                continue
                
            # Get main cast (limit to principal cast)
            cast = [actor['name'] for actor in credits['cast'][:5] if actor.get('order', 10) < 5]
            directors = [crew['name'] for crew in credits['crew'] if crew['job'] == 'Director']
            
            # Record actor collaborations
            for i, actor1 in enumerate(cast):
                for actor2 in cast[i+1:]:
                    actor_collaborations[actor1][actor2] += 1
                    actor_collaborations[actor2][actor1] += 1
                    
            # Record director-actor relationships
            for director in directors:
                for actor in cast:
                    director_actors[director][actor] += 1
    
    return dict(actor_collaborations), dict(director_actors)

def get_box_office_data(movies_df, links_df, max_movies=100):
    """Get box office and rating data for movies"""
    box_office_data = []
    
    # Get sample of movies
    sample_movies = links_df.merge(movies_df, on='movieId').sample(n=min(max_movies, len(movies_df)))
    
    for _, movie in sample_movies.iterrows():
        details = get_movie_details(movie['tmdbId'])
        
        if not details:
            continue
            
        revenue = details.get('revenue', 0)
        budget = details.get('budget', 0)
        vote_average = details.get('vote_average', 0)
        
        # Only include movies with valid data
        if revenue > 0 and budget > 0 and vote_average > 0:
            box_office_data.append({
                'title': movie['title'],
                'revenue': revenue,
                'budget': budget,
                'vote_average': vote_average,
                'genres': movie['genres']
            })
    
    return pd.DataFrame(box_office_data)
