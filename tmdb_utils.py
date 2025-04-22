import requests
import pandas as pd
from collections import defaultdict

TMDB_API_KEY = "YOUR_TMDB_API_KEY"  # Replace with your TMDB API key
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

def build_collaboration_network(movies_df, links_df, max_movies=100):
    """Build actor and director collaboration network"""
    actor_collaborations = defaultdict(lambda: defaultdict(int))
    director_actors = defaultdict(lambda: defaultdict(int))
    
    # Get sample of movies
    sample_movies = links_df.merge(movies_df, on='movieId').sample(n=min(max_movies, len(movies_df)))
    
    for _, movie in sample_movies.iterrows():
        tmdb_id = movie['tmdbId']
        credits = get_movie_credits(tmdb_id)
        
        if not credits:
            continue
            
        # Get main cast and director
        cast = [actor['name'] for actor in credits.get('cast', [])[:5]]  # Top 5 actors
        directors = [crew['name'] for crew in credits.get('crew', []) if crew['job'] == 'Director']
        
        # Record actor collaborations
        for i, actor1 in enumerate(cast):
            for actor2 in cast[i+1:]:
                actor_collaborations[actor1][actor2] += 1
                actor_collaborations[actor2][actor1] += 1
        
        # Record director-actor relationships
        for director in directors:
            for actor in cast:
                director_actors[director][actor] += 1
    
    return actor_collaborations, director_actors

def get_box_office_data(movies_df, links_df, max_movies=100):
    """Get box office and rating data for movies"""
    box_office_data = []
    
    # Get sample of movies
    sample_movies = links_df.merge(movies_df, on='movieId').sample(n=min(max_movies, len(movies_df)))
    
    for _, movie in sample_movies.iterrows():
        details = get_movie_details(movie['tmdbId'])
        
        if not details:
            continue
            
        box_office_data.append({
            'title': movie['title'],
            'revenue': details.get('revenue', 0),
            'budget': details.get('budget', 0),
            'vote_average': details.get('vote_average', 0),
            'genres': movie['genres']
        })
    
    return pd.DataFrame(box_office_data)
