# streamlit_app.py
import streamlit as st
from machine_learning import run_analysis
import pandas as pd

st.set_page_config(page_title="CMPSC 445 - Project 2", layout="wide")
st.title("CMPSC 445 - Project 2: Museum Analysis Dashboard")

st.markdown(
    """
    <div style="margin-top:-8px; margin-bottom:18px;">
      <p style="font-size:18px; color:#6b7280; margin:0; max-width:1200px;">
        By Noam Abraham, Laurence Orji, and Matthew Wybranski. This Streamlit app helps users view analysis of PA museums based on sentiment analysis, ratings, and clustering.
      </p>
      <p style="font-size:18px; color:#6b7280; margin:0; max-width:1200px;">
        To learn more about this project and the code, check out our github <a href="https://github.com/MWybranski/CMPSC445-Project2" target="_blank">here</a>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cache the analysis to avoid rerunning it on every page load
@st.cache_data
def load_analysis_results():
    """Load and cache the analysis results."""
    return run_analysis()

# Load results
with st.spinner("Loading analysis results..."):
    results = load_analysis_results()

# Extract results
working_data = results['working_data']
museum_ranking = results['museum_ranking']
top_10_museums = results['top_10_museums']
cluster_dict = results['cluster_dict']
clustering_plot = results['clustering_plot']
correlation_plot = results['correlation_plot']
least_correlated_museums = results['least_correlated_museums']
cluster_summary = results['cluster_summary']

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Top Museums", "Clustering Analysis", "Sentiment Correlation", "Cluster Details", "Raw Data"]
)

with tab1:
    st.header("Top 10 Museums by Weighted Score")
    st.dataframe(
        top_10_museums.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Complete Museum Rankings")
        st.dataframe(
            museum_ranking.sort_values('weighted_score', ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

with tab2:
    st.header("Museum Clustering Visualization")
    st.markdown("Museums grouped into 4 clusters using K-Means, visualized with PCA dimension reduction.")
    st.pyplot(clustering_plot, use_container_width=True)
    
    st.subheader("Cluster Summary Statistics")
    st.dataframe(cluster_summary, use_container_width=True)

with tab3:
    st.header("Sentiment Analysis vs. User Ratings")
    st.markdown("Correlation between museum average ratings and review sentiment polarity.")
    st.pyplot(correlation_plot, use_container_width=True)
    
    st.subheader("Museums with Lowest Sentiment-Rating Alignment")
    st.markdown("These museums have the biggest discrepancy between their ratings and the sentiment of their reviews.")
    st.dataframe(
        least_correlated_museums[['name', 'museum_average_rating', 'average_polarity', 'residuals']].head(10).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

with tab4:
    st.header("Cluster Characteristics")
    st.markdown("Top categories in each cluster:")
    
    for cluster_id in sorted(cluster_dict.keys()):
        with st.expander(f"Cluster {cluster_id}", expanded=True):
            categories = cluster_dict[cluster_id]
            st.write("Top 10 categories in this cluster:")
            st.dataframe(
                pd.DataFrame({"Category": categories.index, "Count": categories.values}).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

with tab5:
    st.header("Raw Data Browser")
    
    col1, col2 = st.columns(2)
    with col1:
        show_all_reviews = st.checkbox("Show all review data", value=False)
        if show_all_reviews:
            st.subheader("All Reviews with Sentiment Analysis")
            st.dataframe(
                working_data[['name', 'review_text', 'polarity', 'subjectivity', 'museum_average_rating', 'average_polarity']].reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
    
    with col2:
        show_summary = st.checkbox("Show museum summary data", value=False)
        if show_summary:
            st.subheader("Museum Summary")
            st.dataframe(
                working_data[['name', 'museum_average_rating', 'average_polarity', 'review_rating']].drop_duplicates(subset=['name']).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

st.success("Analysis loaded successfully! Use the tabs above to explore different views.")