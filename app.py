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


# Create a collaborative filtering model using cosine similarity
@st.cache_resource
def build_recommendation_model(ratings_df):
    # Create a user-movie matrix
    user_movie_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    
    # Calculate cosine similarity between movies
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    
    # Create a DataFrame with movie similarities
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    
    return movie_similarity_df


# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a feature",
    ["Movie Recommendations", "Genre Analysis", "Rating Distribution",
     "Actor/Director Networks", "Box Office vs. Ratings"]
)

# Movie Recommendations page
if page == "Movie Recommendations":
    st.header("Movie Recommendations")

    if not movies_df.empty and not ratings_df.empty:
        # Build the model
        movie_similarity_df = build_recommendation_model(ratings_df)

        # Get a list of all movie titles
        movie_titles = movies_df['title'].tolist()

        # Let the user select movies they've watched and liked
        st.subheader("Select movies you've enjoyed")
        selected_movies = st.multiselect("Choose movies you like:", movie_titles)

        if selected_movies and st.button("Get Recommendations"):
            # Get the movieIds for the selected movies
            selected_movie_ids = movies_df[movies_df['title'].isin(selected_movies)]['movieId'].tolist()

            if selected_movie_ids:
                # Get similar movies
                similar_movies = []
                for movie_id in selected_movie_ids:
                    if movie_id in movie_similarity_df.index:
                        similar = movie_similarity_df[movie_id].sort_values(ascending=False)[1:11]
                        similar_movies.extend(list(zip([movie_id] * len(similar), similar.index, similar.values)))

                # Sort by similarity and get unique recommendations
                similar_movies.sort(key=lambda x: x[2], reverse=True)
                recommended_ids = []
                for _, movie_id, _ in similar_movies:
                    if movie_id not in selected_movie_ids and movie_id not in recommended_ids:
                        recommended_ids.append(movie_id)
                        if len(recommended_ids) == 10:
                            break

                # Display recommendations
                st.subheader("Your Personalized Recommendations:")
                for i, movie_id in enumerate(recommended_ids, 1):
                    movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
                    st.write(f"{i}. **{movie_info['title']}**")
                    st.write(f"   Genres: {movie_info['genres'].replace('|', ', ')}")
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

    st.info("""
    For a complete implementation, this would require additional data about actors and directors, 
    which isn't included in the basic MovieLens dataset. 

    In a full implementation, we would:
    1. Use the IMDb IDs from the links.csv file to fetch actor and director information
    2. Build networks of actors who have worked together
    3. Identify directors who frequently work with certain actors
    4. Visualize these networks using NetworkX

    Below is a simplified example visualization of how such a network might look.
    """)

    # Create a simple example network
    G = nx.barabasi_albert_graph(30, 2)

    # Generate some example actor names
    actors = ["Actor " + str(i + 1) for i in range(30)]

    # Assign names to nodes
    node_labels = {i: actors[i] for i in range(30)}

    # Draw the network
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, with_labels=True, node_color='skyblue',
                     node_size=500, labels=node_labels, font_size=8, ax=ax)
    ax.set_title('Example Actor Collaboration Network')
    ax.axis('off')

    st.pyplot(fig)

    st.write("""
    This simplified visualization represents how actors might be connected through movie collaborations. 
    The full implementation would use real data to show:

    - Which actors frequently work together
    - Which directors work with specific actors repeatedly
    - Communities of actors who often collaborate
    - The "six degrees of separation" phenomenon in the film industry
    """)

# Box Office vs. Ratings page
elif page == "Box Office vs. Ratings":
    st.header("Box Office Performance vs. Critic Scores")

    st.info("""
    The MovieLens dataset doesn't include box office information or critic scores.
    To implement this feature, we would need to:

    1. Use the IMDb IDs to fetch box office data and critic scores from additional sources
    2. Analyze the correlation between financial success and critical reception
    3. Visualize these relationships across different genres and time periods

    Below is a simplified example of how such an analysis might look.
    """)

    # Create some sample data
    np.random.seed(42)
    n_movies = 100

    sample_data = pd.DataFrame({
        'Movie': [f"Movie {i + 1}" for i in range(n_movies)],
        'Box_Office': np.random.exponential(scale=100, size=n_movies),
        'Critic_Score': np.random.normal(loc=7, scale=1.5, size=n_movies),
        'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'], size=n_movies)
    })

    # Add some correlation
    sample_data['Box_Office'] = sample_data['Box_Office'] + sample_data['Critic_Score'] * 5 + np.random.normal(0, 20,
                                                                                                               n_movies)

    # Create interactive scatter plot with plotly
    fig = px.scatter(sample_data,
                    x='Critic_Score',
                    y='Box_Office',
                    color='Genre',
                    title='Example: Box Office Revenue vs. Critic Scores by Genre',
                    labels={'Critic_Score': 'Critic Score (0-10)',
                           'Box_Office': 'Box Office Revenue (millions $)',
                           'Genre': 'Genre'},
                    opacity=0.7)

    # Add trend line
    fig.add_trace(go.Scatter(
        x=sample_data['Critic_Score'],
        y=np.poly1d(np.polyfit(sample_data['Critic_Score'], sample_data['Box_Office'], 1))(sample_data['Critic_Score']),
        mode='lines',
        name='Trend',
        line=dict(color='black'),
        showlegend=True
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    This example chart illustrates how we might visualize the relationship between box office performance and critic scores.
    The actual implementation would analyze real data to answer questions like:

    - Do critically acclaimed movies perform better at the box office?
    - Which genres show the strongest correlation between reviews and financial success?
    - Are there notable outliers (critically panned blockbusters or acclaimed box office failures)?
    - How has this relationship changed over time?
    """)

# Footer
st.markdown("---")
st.markdown("Movie Recommendation Engine - Built with Streamlit, Surprise, and NetworkX")