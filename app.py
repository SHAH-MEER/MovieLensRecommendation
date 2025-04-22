import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import requests
import io
import zipfile
from tmdb_utils import build_collaboration_network, get_box_office_data
from recommender_utils import build_advanced_recommender, get_recommendations, get_cold_start_recommendations, get_genre_preferences, get_recommendation_explanation
from movie_utils import search_movies, display_movie_details

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Movie Recommendation Engine")
st.markdown("""
This application provides movie recommendations based on user preferences and offers various movie data analyses.
Use the sidebar to navigate between different features.
""")


# Function to load MovieLens data
@st.cache_data
def load_movielens_data():
    # Download MovieLens 100K dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    try:
        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))

        # Extract and read the relevant files
        movies_df = pd.read_csv(z.open('ml-latest-small/movies.csv'))
        ratings_df = pd.read_csv(z.open('ml-latest-small/ratings.csv'))
        links_df = pd.read_csv(z.open('ml-latest-small/links.csv'))
        tags_df = pd.read_csv(z.open('ml-latest-small/tags.csv'))

        return movies_df, ratings_df, links_df, tags_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide sample data if download fails
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# Load data
movies_df, ratings_df, links_df, tags_df = load_movielens_data()


# Process genre data
def extract_genres(movies_df):
    # Create a dictionary to store genre counts
    genre_counts = defaultdict(int)

    # Extract genres from each movie
    for genres in movies_df['genres'].str.split('|'):
        for genre in genres:
            if genre != '(no genres listed)':
                genre_counts[genre] += 1

    # Convert to DataFrame
    genre_df = pd.DataFrame({
        'Genre': list(genre_counts.keys()),
        'Count': list(genre_counts.values())
    }).sort_values('Count', ascending=False)

    return genre_df


# Create a collaborative filtering model using advanced methods
@st.cache_resource
def build_recommendation_model(ratings_df):
    with st.spinner("Building recommendation model... This may take a moment."):
        return build_advanced_recommender(ratings_df)


# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a feature",
    ["Movie Recommendations", "Genre Analysis", "Rating Distribution",
     "Actor/Director Networks", "Box Office vs. Ratings", "Movie Details"]
)

# Add Search to sidebar
st.sidebar.markdown("---")
search_query = st.sidebar.text_input("Search Movies")
if search_query:
    search_results = search_movies(search_query, movies_df)
    st.sidebar.write(f"Found {len(search_results)} results")
    for _, movie in search_results.iterrows():
        if st.sidebar.button(movie['title']):
            st.session_state.selected_movie = movie
            st.session_state.page = "Movie Details"

# Update page selection to include Movie Details
if hasattr(st.session_state, 'page'):
    page = st.session_state.page

# Movie Details page
if page == "Movie Details":
    if hasattr(st.session_state, 'selected_movie'):
        movie = st.session_state.selected_movie
        tmdb_id = links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId']
        display_movie_details(movie, tmdb_id)

# Movie Recommendations page
elif page == "Movie Recommendations":
    st.header("Movie Recommendations")

    if not movies_df.empty and not ratings_df.empty:
        # Build the model
        similarity_matrix = build_recommendation_model(ratings_df)

        # Get a list of all movie titles
        movie_titles = movies_df['title'].tolist()

        # Let the user select movies they've watched and liked
        st.subheader("Select movies you've enjoyed")
        selected_movies = st.multiselect("Choose movies you like:", movie_titles)

        if not selected_movies:
            # Cold start recommendations
            st.subheader("Popular Movies You Might Like")
            genre_pref = None
            if 'genre_preferences' in st.session_state:
                genre_pref = st.session_state.genre_preferences
            recommendations = get_cold_start_recommendations(movies_df, ratings_df, genre_pref)

            for _, movie in recommendations.iterrows():
                col1, col2 = st.columns([1, 3])
                details = get_movie_details(links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId'])

                with col1:
                    poster_url = get_movie_poster_url(details)
                    if poster_url:
                        st.image(poster_url, width=100)

                with col2:
                    st.write(f"**{movie['title']}**")
                    st.write(f"Genres: {movie['genres']}")

        elif selected_movies and st.button("Get Recommendations"):
            with st.spinner("Finding recommendations..."):
                # Get the movieIds for the selected movies
                selected_movie_ids = movies_df[movies_df['title'].isin(selected_movies)]['movieId'].tolist()

                if selected_movie_ids:
                    # Get recommendations
                    recommendations = get_recommendations(
                        movies_df, 
                        similarity_matrix, 
                        selected_movie_ids
                    )

                    # Display recommendations
                    st.subheader("Your Personalized Recommendations:")
                    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                        col1, col2 = st.columns([1, 3])
                        details = get_movie_details(links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId'])

                        with col1:
                            poster_url = get_movie_poster_url(details)
                            if poster_url:
                                st.image(poster_url, width=100)

                        with col2:
                            st.write(f"{i}. **{movie['title']}**")
                            st.write(f"Genres: {movie['genres'].replace('|', ', ')}")
                            st.write(f"Similarity Score: {movie['similarity_score']:.2f}")
                            explanation = get_recommendation_explanation(movie, movie['similarity_score'], 
                                                                      movies_df, selected_movies)
                            st.write(f"*Why this recommendation?* {explanation}")
                else:
                    st.warning("Could not find the selected movies in our database.")

        # Display sample of available movies
        st.subheader("Sample of Available Movies")
        st.dataframe(movies_df[['title', 'genres']].sample(10))
    else:
        st.error("Failed to load movie data. Please try refreshing the page.")

# Genre Analysis page
elif page == "Genre Analysis":
    st.header("Genre Distribution Analysis")

    if not movies_df.empty:
        # Process genre data
        genre_df = extract_genres(movies_df)

        # Create interactive bar chart with plotly
        fig = px.bar(genre_df, 
                    x='Count', 
                    y='Genre',
                    title='Movie Count by Genre',
                    labels={'Count': 'Number of Movies', 'Genre': 'Genre'},
                    color='Genre')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show genre data
        st.subheader("Genre Data")
        st.dataframe(genre_df)

        # Genre combinations analysis
        st.subheader("Popular Genre Combinations")

        # Get combinations of genres
        genre_combinations = movies_df['genres'].value_counts().head(10)

        # Display as a table
        st.table(pd.DataFrame({
            'Genre Combination': genre_combinations.index,
            'Number of Movies': genre_combinations.values
        }))
    else:
        st.error("Failed to load movie data. Please try refreshing the page.")

# Rating Distribution page
elif page == "Rating Distribution":
    st.header("Rating Distribution Analysis")

    if not ratings_df.empty and not movies_df.empty:
        # Merge ratings with movie information
        merged_df = pd.merge(ratings_df, movies_df, on='movieId')

        # Distribution of ratings using plotly
        fig = px.histogram(merged_df, 
                          x='rating',
                          nbins=9,
                          title='Distribution of Movie Ratings',
                          labels={'rating': 'Rating', 'count': 'Frequency'})
        fig.add_trace(go.Histogram(x=merged_df['rating'], nbinsx=9, name='Count'))
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        # Average rating by genre
        st.subheader("Average Rating by Genre")

        # Explode the genres column to get one row per genre
        genre_ratings = merged_df.copy()
        genre_ratings['genres'] = genre_ratings['genres'].str.split('|')
        genre_ratings = genre_ratings.explode('genres')

        # Filter out rows with no genre
        genre_ratings = genre_ratings[genre_ratings['genres'] != '(no genres listed)']

        # Calculate average ratings by genre
        avg_ratings = genre_ratings.groupby('genres')['rating'].agg(['mean', 'count']).reset_index()
        avg_ratings = avg_ratings.sort_values('mean', ascending=False)

        # Plot average ratings with plotly
        fig = px.bar(avg_ratings,
                    x='mean',
                    y='genres',
                    title='Average Rating by Genre',
                    labels={'mean': 'Average Rating', 'genres': 'Genre'},
                    color='genres',
                    text=[f"n={c}" for c in avg_ratings['count']])
        fig.update_layout(showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Top rated movies
        st.subheader("Top Rated Movies (with at least 100 ratings)")

        # Calculate the number of ratings per movie
        movie_ratings_count = merged_df.groupby('movieId')['rating'].count().reset_index()
        movie_ratings_count.columns = ['movieId', 'ratings_count']

        # Calculate the average rating per movie
        movie_ratings_avg = merged_df.groupby('movieId')['rating'].mean().reset_index()
        movie_ratings_avg.columns = ['movieId', 'rating_avg']

        # Merge count and average
        movie_ratings = pd.merge(movie_ratings_count, movie_ratings_avg, on='movieId')

        # Merge with movie information
        movie_ratings = pd.merge(movie_ratings, movies_df, on='movieId')

        # Filter to movies with at least 100 ratings
        popular_movies = movie_ratings[movie_ratings['ratings_count'] >= 100].sort_values('rating_avg', ascending=False)

        # Display top 10
        st.table(popular_movies[['title', 'rating_avg', 'ratings_count', 'genres']].head(10))
    else:
        st.error("Failed to load movie data. Please try refreshing the page.")

# Actor/Director Networks page
elif page == "Actor/Director Networks":
    st.header("Actor/Director Network Analysis")
    
    if not movies_df.empty and not links_df.empty:
        with st.spinner("Fetching collaboration data..."):
            actor_collaborations, director_actors = build_collaboration_network(movies_df, links_df)
            
            # Create network graph
            G = nx.Graph()
            
            # Add actor collaboration edges
            for actor1, collaborators in actor_collaborations.items():
                for actor2, weight in collaborators.items():
                    if weight > 1:  # Only show frequent collaborations
                        G.add_edge(actor1, actor2, weight=weight)
            
            # Create the visualization
            fig, ax = plt.subplots(figsize=(15, 15))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title("Actor Collaboration Network")
            st.pyplot(fig)
            
            # Show director insights
            st.subheader("Director-Actor Frequent Collaborations")
            for director, actors in director_actors.items():
                frequent_collabs = {actor: count for actor, count in actors.items() if count > 1}
                if frequent_collabs:
                    st.write(f"**{director}** frequently works with:")
                    for actor, count in sorted(frequent_collabs.items(), key=lambda x: x[1], reverse=True)[:3]:
                        st.write(f"- {actor} ({count} collaborations)")
    else:
        st.error("Failed to load movie data.")

# Box Office vs. Ratings page
elif page == "Box Office vs. Ratings":
    st.header("Box Office Performance vs. Critic Scores")
    
    if not movies_df.empty and not links_df.empty:
        with st.spinner("Fetching box office data..."):
            box_office_df = get_box_office_data(movies_df, links_df)
            
            if not box_office_df.empty:
                # Preprocess data
                box_office_df['revenue_millions'] = box_office_df['revenue'] / 1_000_000
                # Extract primary genre
                box_office_df['primary_genre'] = box_office_df['genres'].str.split('|').str[0]
                
                # Create scatter plot
                fig = px.scatter(box_office_df,
                               x='vote_average',
                               y='revenue_millions',
                               color='primary_genre',
                               title='Box Office Revenue vs. Rating',
                               labels={'vote_average': 'TMDB Rating',
                                     'revenue_millions': 'Box Office Revenue (Millions $)',
                                     'primary_genre': 'Primary Genre'},
                               hover_data={
                                   'title': True,
                                   'revenue_millions': ':.1f',
                                   'vote_average': ':.1f'
                               })
                
                # Update layout
                fig.update_layout(
                    xaxis_range=[0, 10],
                    yaxis_range=[0, box_office_df['revenue_millions'].max() * 1.1],
                    height=600
                )
                
                # Add trend line
                z = np.polyfit(box_office_df['vote_average'], box_office_df['revenue_millions'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=box_office_df['vote_average'],
                    y=p(box_office_df['vote_average']),
                    mode='lines',
                    name='Trend',
                    line=dict(color='black', dash='dash'),
                    showlegend=True
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show ROI analysis
                st.subheader("Return on Investment Analysis")
                box_office_df['roi'] = box_office_df.apply(
                    lambda x: ((x['revenue'] - x['budget']) / x['budget']) if x['budget'] > 0 else 0, 
                    axis=1
                )
                box_office_df['roi_pct'] = box_office_df['roi'].apply(lambda x: f"{x*100:.1f}%")
                
                best_roi = box_office_df.nlargest(5, 'roi')[
                    ['title', 'roi_pct', 'vote_average', 'primary_genre']
                ]
                best_roi.columns = ['Movie Title', 'Return on Investment', 'Rating', 'Genre']
                st.table(best_roi)
            else:
                st.error("No box office data available.")
    else:
        st.error("Failed to load movie data.")

# Footer
st.markdown("---")
st.markdown("Movie Recommendation Engine - Built with Streamlit, Surprise, and NetworkX")