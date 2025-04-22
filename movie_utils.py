import streamlit as st
import pandas as pd
from tmdb_utils import get_movie_details, get_movie_credits

def search_movies(query, movies_df):
    """Search movies by title, genres, or year"""
    query = query.lower()
    mask = (
        movies_df['title'].str.lower().str.contains(query) |
        movies_df['genres'].str.lower().str.contains(query)
    )
    return movies_df[mask]

def get_movie_poster_url(movie_details):
    """Get movie poster URL from TMDB"""
    if movie_details and movie_details.get('poster_path'):
        return f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
    return None

def display_movie_details(movie_info, tmdb_id):
    """Display detailed movie information"""
    details = get_movie_details(tmdb_id)
    credits = get_movie_credits(tmdb_id)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        poster_url = get_movie_poster_url(details)
        if poster_url:
            st.image(poster_url, width=300)
            
    with col2:
        st.subheader(movie_info['title'])
        st.write(f"**Genres:** {movie_info['genres']}")
        
        if details:
            st.write(f"**Release Date:** {details.get('release_date', 'N/A')}")
            st.write(f"**Rating:** {details.get('vote_average', 'N/A')}/10")
            st.write(f"**Overview:** {details.get('overview', 'N/A')}")
        
        if credits:
            directors = [c['name'] for c in credits.get('crew', []) if c['job'] == 'Director']
            cast = [c['name'] for c in credits.get('cast', [])[:5]]
            
            if directors:
                st.write(f"**Director(s):** {', '.join(directors)}")
            if cast:
                st.write(f"**Main Cast:** {', '.join(cast)}")
