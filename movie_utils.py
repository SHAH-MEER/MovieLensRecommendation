import streamlit as st
import pandas as pd
from tmdb_utils import get_movie_details, get_movie_credits, get_movie_poster_url

def search_movies(query, movies_df):
    """Search movies by title or genres with improved partial matching"""
    if not query:
        return pd.DataFrame()
    
    query = query.lower()
    # Create mask for partial matches in title or genres
    mask = (
        movies_df['title'].str.lower().str.contains(query, na=False) |
        movies_df['genres'].str.lower().str.contains(query, na=False)
    )
    return movies_df[mask].head(20)  # Limit results to prevent performance issues

def display_search_result(movie, movie_details, index):
    """Display a single movie search result with poster and details"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster_url = get_movie_poster_url(movie_details)
        if poster_url:
            st.image(poster_url, width=150)
    
    with col2:
        st.markdown(f"### {movie['title']}")
        st.write(f"**Genres:** {movie['genres'].replace('|', ', ')}")
        if movie_details:
            if movie_details.get('overview'):
                st.write(movie_details['overview'][:200] + "..." if len(movie_details['overview']) > 200 else movie_details['overview'])
            if movie_details.get('vote_average'):
                st.write(f"**Rating:** ⭐ {movie_details['vote_average']}/10")
        
        if st.button("View Details", key=f"view_{index}"):
            return True
    return False

def display_movie_details(movie_info, tmdb_id):
    """Display detailed movie information"""
    details = get_movie_details(tmdb_id)
    credits = get_movie_credits(tmdb_id)
    
    if details and credits:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            poster_url = get_movie_poster_url(details)
            if poster_url:
                st.image(poster_url, width=300)
            
        with col2:
            st.title(movie_info['title'])
            if details.get('tagline'):
                st.write(f"*{details['tagline']}*")
            
            st.markdown("### Movie Info")
            st.write(f"**Release Date:** {details.get('release_date', 'N/A')}")
            st.write(f"**Rating:** {details.get('vote_average', 'N/A')}/10 ({details.get('vote_count', 0)} votes)")
            st.write(f"**Genres:** {movie_info['genres']}")
            
            if details.get('overview'):
                st.markdown("### Overview")
                st.write(details['overview'])
            
            # Cast and Crew
            st.markdown("### Cast & Crew")
            directors = [c['name'] for c in credits.get('crew', []) if c['job'] == 'Director']
            if directors:
                st.write(f"**Director(s):** {', '.join(directors)}")
            
            cast = [c['name'] for c in credits.get('cast', [])[:5]]
            if cast:
                st.write(f"**Top Cast:** {', '.join(cast)}")
            
            # Additional Details
            if details.get('budget') or details.get('revenue'):
                st.markdown("### Box Office")
                if details.get('budget'):
                    st.write(f"**Budget:** ${details['budget']:,}")
                if details.get('revenue'):
                    st.write(f"**Revenue:** ${details['revenue']:,}")
    else:
        st.error("Could not fetch movie details. Please try again later.")
