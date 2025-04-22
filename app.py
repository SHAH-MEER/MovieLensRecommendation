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
from tmdb_utils import (
    build_collaboration_network, 
    get_box_office_data,
    get_movie_details,
    get_movie_poster_url
)
from recommender_utils import (
    build_advanced_recommender, 
    get_recommendations, 
    get_cold_start_recommendations, 
    get_genre_preferences, 
    get_recommendation_explanation
)
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


@st.cache_data
def load_movielens_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    try:
        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        movies_df = pd.read_csv(z.open('ml-latest-small/movies.csv'))
        ratings_df = pd.read_csv(z.open('ml-latest-small/ratings.csv'))
        links_df = pd.read_csv(z.open('ml-latest-small/links.csv'))
        tags_df = pd.read_csv(z.open('ml-latest-small/tags.csv'))
        return movies_df, ratings_df, links_df, tags_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


movies_df, ratings_df, links_df, tags_df = load_movielens_data()


def extract_genres(movies_df):
    genre_counts = defaultdict(int)
    for genres in movies_df['genres'].str.split('|'):
        for genre in genres:
            if genre != '(no genres listed)':
                genre_counts[genre] += 1
    genre_df = pd.DataFrame({
        'Genre': list(genre_counts.keys()),
        'Count': list(genre_counts.values())
    }).sort_values('Count', ascending=False)
    return genre_df


@st.cache_resource
def build_recommendation_model(ratings_df):
    with st.spinner("Building recommendation model... This may take a moment."):
        return build_advanced_recommender(ratings_df)


if 'current_page' not in st.session_state:
    st.session_state.current_page = "Movie Recommendations"
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'previous_page' not in st.session_state:
    st.session_state.previous_page = None
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""
if 'clear_search' not in st.session_state:
    st.session_state.clear_search = False


def view_movie_details(movie):
    st.session_state.previous_page = st.session_state.current_page
    st.session_state.current_page = "Movie Details"
    st.session_state.selected_movie = movie
    st.rerun()


def go_back():
    st.session_state.current_page = st.session_state.previous_page
    st.session_state.selected_movie = None
    st.rerun()


def clear_search():
    st.session_state.selected_movie = None
    st.session_state.clear_search = True
    st.session_state.last_search = ""


def plot_interactive_network(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'),
            go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text)
        ],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig


def display_movie_card(movie, details, poster_url):
    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            if poster_url:
                st.image(poster_url, width=150)
        with col2:
            st.markdown(f"### {movie['title']}")
            st.markdown(f"**Rating:** {'⭐' * int(details.get('vote_average', 0)/2)}")
            st.markdown(f"**Genres:** {movie['genres'].replace('|', ', ')}")
            if details and details.get('overview'):
                st.markdown(details['overview'][:200] + "...")
        st.markdown('</div>', unsafe_allow_html=True)


st.sidebar.title("Navigation")
main_page = st.sidebar.radio(
    "Select a feature",
    ["Search Movies", "Movie Recommendations", "Genre Analysis", "Rating Distribution",
     "Actor/Director Networks", "Box Office vs. Ratings"]
)

if main_page != "Movie Details":
    st.session_state.current_page = main_page


if st.session_state.current_page == "Search Movies":
    st.header("Search Movies")
    search_col1, search_col2, _ = st.columns([1, 2, 1])
    with search_col2:
        search_query = st.text_input(
            "Search by title or genre",
            value="" if st.session_state.clear_search else st.session_state.last_search,
            key="search_input",
            placeholder="Enter movie title or genre..."
        )
        col1, col2 = st.columns(2)
        with col1:
            search_clicked = st.button("🔍 Search", use_container_width=True)
        with col2:
            if st.button("🔄 Clear", type="secondary", use_container_width=True):
                clear_search()
                st.rerun()
    if search_query:
        st.session_state.last_search = search_query
        search_results = search_movies(search_query, movies_df)
        if not search_results.empty:
            st.subheader(f"Found {len(search_results)} movies")
            for i, (_, movie) in enumerate(search_results.iterrows()):
                st.markdown("---")
                tmdb_id = links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId']
                details = get_movie_details(tmdb_id)
                poster_url = get_movie_poster_url(details)
                display_movie_card(movie, details, poster_url)
                if st.button("View Details →", key=f"view_{movie['movieId']}", use_container_width=True):
                    view_movie_details(movie)
        else:
            st.info("No movies found matching your search.")

elif st.session_state.current_page == "Movie Details":
    if st.button("← Back", key="back_button"):
        go_back()
    if st.session_state.selected_movie is not None:
        movie = st.session_state.selected_movie
        tmdb_id = links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId']
        display_movie_details(movie, tmdb_id)
    else:
        st.error("No movie selected")
        go_back()

elif st.session_state.current_page == "Movie Recommendations":
    st.header("Movie Recommendations")
    if not movies_df.empty and not ratings_df.empty:
        similarity_matrix = build_recommendation_model(ratings_df)
        movie_titles = movies_df['title'].tolist()
        st.subheader("Select movies you've enjoyed")
        selected_movies = st.multiselect("Choose movies you like:", movie_titles)
        if not selected_movies:
            st.subheader("Popular Movies You Might Like")
            genre_pref = st.session_state.get('genre_preferences', None)
            recommendations = get_cold_start_recommendations(movies_df, ratings_df, genre_pref)
            for _, movie in recommendations.iterrows():
                tmdb_id = links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId']
                details = get_movie_details(tmdb_id)
                poster_url = get_movie_poster_url(details)
                display_movie_card(movie, details, poster_url)
                if st.button("View Details →", key=f"view_{movie['movieId']}", use_container_width=True):
                    view_movie_details(movie)
        elif selected_movies and st.button("Get Recommendations"):
            with st.spinner("Finding recommendations..."):
                selected_movie_ids = movies_df[movies_df['title'].isin(selected_movies)]['movieId'].tolist()
                if selected_movie_ids:
                    recommendations = get_recommendations(movies_df, similarity_matrix, selected_movie_ids)
                    st.subheader("Your Personalized Recommendations:")
                    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                        details = get_movie_details(links_df[links_df['movieId'] == movie['movieId']].iloc[0]['tmdbId'])
                        poster_url = get_movie_poster_url(details)
                        display_movie_card(movie, details, poster_url)
                        st.write(f"Similarity Score: {movie['similarity_score']:.2f}")
                        explanation = get_recommendation_explanation(movie, movie['similarity_score'], 
                                                                      movies_df, selected_movies)
                        st.write(f"*Why this recommendation?* {explanation}")
                else:
                    st.warning("Could not find the selected movies in our database.")
        st.subheader("Sample of Available Movies")
        st.dataframe(movies_df[['title', 'genres']].sample(10))
    else:
        st.error("Failed to load movie data. Please try refreshing the page.")

elif st.session_state.current_page == "Genre Analysis":
    st.header("Genre Distribution Analysis")
    if not movies_df.empty:
        genre_df = extract_genres(movies_df)
        fig = px.bar(genre_df, 
                    x='Count', 
                    y='Genre',
                    title='Movie Count by Genre',
                    labels={'Count': 'Number of Movies', 'Genre': 'Genre'},
                    color='Genre')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Genre Data")
        st.dataframe(genre_df)
        st.subheader("Popular Genre Combinations")
        genre_combinations = movies_df['genres'].value_counts().head(10)
        st.table(pd.DataFrame({
            'Genre Combination': genre_combinations.index,
            'Number of Movies': genre_combinations.values
        }))
    else:
        st.error("Failed to load movie data. Please try refreshing the page.")

elif st.session_state.current_page == "Rating Distribution":
    st.header("Rating Distribution Analysis")
    if not ratings_df.empty and not movies_df.empty:
        merged_df = pd.merge(ratings_df, movies_df, on='movieId')
        fig = px.histogram(merged_df, 
                          x='rating',
                          nbins=9,
                          title='Distribution of Movie Ratings',
                          labels={'rating': 'Rating', 'count': 'Frequency'})
        fig.add_trace(go.Histogram(x=merged_df['rating'], nbinsx=9, name='Count'))
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Average Rating by Genre")
        genre_ratings = merged_df.copy()
        genre_ratings['genres'] = genre_ratings['genres'].str.split('|')
        genre_ratings = genre_ratings.explode('genres')
        genre_ratings = genre_ratings[genre_ratings['genres'] != '(no genres listed)']
        avg_ratings = genre_ratings.groupby('genres')['rating'].agg(['mean', 'count']).reset_index()
        avg_ratings = avg_ratings.sort_values('mean', ascending=False)
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
        st.subheader("Top Rated Movies (with at least 100 ratings)")
        movie_ratings_count = merged_df.groupby('movieId')['rating'].count().reset_index()
        movie_ratings_count.columns = ['movieId', 'ratings_count']
        movie_ratings_avg = merged_df.groupby('movieId')['rating'].mean().reset_index()
        movie_ratings_avg.columns = ['movieId', 'rating_avg']
        movie_ratings = pd.merge(movie_ratings_count, movie_ratings_avg, on='movieId')
        movie_ratings = pd.merge(movie_ratings, movies_df, on='movieId')
        popular_movies = movie_ratings[movie_ratings['ratings_count'] >= 100].sort_values('rating_avg', ascending=False)
        st.table(popular_movies[['title', 'rating_avg', 'ratings_count', 'genres']].head(10))
    else:
        st.error("Failed to load movie data. Please try refreshing the page.")

elif st.session_state.current_page == "Actor/Director Networks":
    st.header("Actor/Director Network Analysis")
    if not movies_df.empty and not links_df.empty:
        actor_collaborations, director_actors = build_collaboration_network(movies_df, links_df)
        G = nx.Graph()
        for actor1, collaborators in actor_collaborations.items():
            for actor2, weight in collaborators.items():
                if weight >= 1:
                    G.add_edge(actor1, actor2, weight=weight)
        if len(G.nodes()) > 0:
            fig, ax = plt.subplots(figsize=(15, 15))
            pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
            edge_weights = [G[u][v]['weight']*2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights)
            centrality = nx.degree_centrality(G)
            node_sizes = [centrality[node] * 5000 + 100 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=node_sizes,
                                 alpha=0.7)
            important_nodes = {node: node for node, size in zip(G.nodes(), node_sizes) if size > 200}
            nx.draw_networkx_labels(G, pos, labels=important_nodes, font_size=8)
            plt.title("Actor Collaboration Network\n(Node size indicates collaboration frequency)")
            ax.axis('off')
            st.pyplot(fig)
            st.subheader("Key Collaboration Insights")
            degrees = dict(G.degree())
            top_actors = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            st.write("Most Collaborative Actors:")
            for actor, degree in top_actors:
                st.write(f"- {actor} ({degree} collaborations)")
            st.subheader("Notable Director-Actor Partnerships")
            for director, actors in list(director_actors.items())[:5]:
                frequent_collabs = {actor: count for actor, count in actors.items() if count >= 1}
                if frequent_collabs:
                    st.write(f"**{director}** frequently works with:")
                    for actor, count in sorted(frequent_collabs.items(), key=lambda x: x[1], reverse=True)[:3]:
                        st.write(f"- {actor} ({count} collaboration{'s' if count > 1 else ''})")
        else:
            st.warning("Building collaboration network... Please wait or refresh the page.")

elif st.session_state.current_page == "Box Office vs. Ratings":
    st.header("Box Office Performance vs. Critic Scores")
    if not movies_df.empty and not links_df.empty:
        with st.spinner("Fetching box office data..."):
            box_office_df = get_box_office_data(movies_df, links_df)
            if not box_office_df.empty:
                box_office_df['revenue_millions'] = box_office_df['revenue'] / 1_000_000
                box_office_df['primary_genre'] = box_office_df['genres'].str.split('|').str[0]
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
                fig.update_layout(
                    xaxis_range=[0, 10],
                    yaxis_range=[0, box_office_df['revenue_millions'].max() * 1.1],
                    height=600
                )
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

st.markdown("---")
st.markdown("Movie Recommendation Engine - Built with Streamlit, Surprise, and NetworkX")