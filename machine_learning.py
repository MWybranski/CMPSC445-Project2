import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob


def run_analysis():
    """
    Run the complete museum analysis pipeline and return all results.
    
    Returns:
        dict: A dictionary containing:
            - working_data: DataFrame with all reviews and sentiment analysis
            - museum_ranking: DataFrame with museums ranked by weighted score
            - top_10_museums: Top 10 museums by weighted score
            - cluster_dict: Dictionary mapping cluster IDs to top categories
            - clustering_plot: Matplotlib figure of the PCA clustering visualization
            - correlation_plot: Matplotlib figure of sentiment vs. rating correlation
            - least_correlated_museums: Museums with largest sentiment/rating discrepancies
            - cluster_summary: Summary statistics for each cluster
    """
    
    ##################
    # Preprocessing
    ##################

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Read csv files
    overview_df = pd.read_csv('datasets/overview.csv')
    reviews_df = pd.read_csv('datasets/detailed_reviews.csv', low_memory=False)

    ## Dataset cleaning - overview_df
    # Drop rows with NaN value in "categories" column
    overview_df = overview_df.dropna(subset=['categories'])

    # Filter the dataset for museum data only
    # Check row-by-row if value under "categories" contains "museum"
    museum_df = overview_df[overview_df['categories'].str.contains('museum', case=False)]

    # Select only relevant columns: name, reviews, rating, categories, address
    museum_df = museum_df[['name', 'reviews', 'rating', 'categories', 'address']]

    ## Dataset cleaning - reviews_df
    # Select only rows that have value "en" in the original_language column
    reviews_df = reviews_df[reviews_df['original_language'] == 'en']

    # Select only relevant columns: place_name, rating, review_text
    reviews_df = reviews_df[['place_name', 'rating', 'review_text']]

    # Combine the datasets using name/place_name
    master_df = pd.merge(museum_df, reviews_df, left_on='name', right_on='place_name', how='inner')
    master_df["museum_average_rating"] = master_df["rating_x"]
    master_df["review_rating"] = master_df["rating_y"]
    master_df.drop(columns=['rating_x', 'rating_y', 'place_name'], inplace=True)

    # Drop rows with no value in review_text
    master_df.dropna(subset=['review_text'], inplace=True)

    # Review text preprocessing
    lemma = WordNetLemmatizer()

    working_data = master_df.copy()

    working_data['review_text'] = working_data['review_text'].str.lower()  # Lowercasing
    working_data['review_text'] = working_data['review_text'].replace(r'[^a-z\s]', ' ', regex=True)  # Removing punctuation and numbers

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN # Default to noun if no clear mapping

    preprocessed_records = []

    for record in working_data['review_text']:
        record = str(record) # Convert record to string
        tokenized_record = nltk.word_tokenize(record) # Tokenization
        # Removing stopwords
        filtered_record = [word for word in tokenized_record if word not in stop_words]
        # POS tagging
        pos_tagged_record = nltk.pos_tag(filtered_record)
        # Lemmatization with POS
        lemmatized_record = [lemma.lemmatize(word, get_wordnet_pos(pos_tag)) for word, pos_tag in pos_tagged_record]
        new_record = " ".join(lemmatized_record)
        preprocessed_records.append(new_record)

    working_data['review_text'] = preprocessed_records

    #######################
    # Sentiment analysis
    #######################
    working_data['polarity'] = working_data['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    working_data['subjectivity'] = working_data['review_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    ## hted ranking implementation
    # Calculate average polarity for each museum
    average_polarity = working_data.groupby('name')['polarity'].mean().reset_index()
    average_polarity.rename(columns={'polarity': 'average_polarity'}, inplace=True)

    # Merge the average polarity with the original DataFrame
    working_data = pd.merge(working_data, average_polarity, on='name', how='left')

    # Normalize features using scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(working_data[['average_polarity', 'museum_average_rating', 'reviews']])
    scaled_df = pd.DataFrame(scaled_features, columns=['sc_average_polarity', 'sc_museum_average_rating', 'sc_number_of_reviews'])

    # Merge scaled_df and working_data into a new DataFrame, replacing previous metrics with new scaled ones
    scaled_master_df = pd.concat([working_data.drop(['average_polarity', 'museum_average_rating', 'reviews'], axis=1), scaled_df], axis=1)

    # Define weights for the scaled features
    # sc_average_polarity is weighted highest, then sc_number_of_reviews, and finally sc_museum_average_rating
    weights = {
        'sc_average_polarity': 0.5,
        'sc_number_of_reviews': 0.3,
        'sc_museum_average_rating': 0.2
    }

    # Calculate the weighted score
    scaled_master_df['weighted_score'] = (
        scaled_master_df['sc_average_polarity'] * weights['sc_average_polarity'] +
        scaled_master_df['sc_number_of_reviews'] * weights['sc_number_of_reviews'] +
        scaled_master_df['sc_museum_average_rating'] * weights['sc_museum_average_rating']
    )

    # Group by museum name and calculate the mean of the weighted_score to get one score per museum
    museum_ranking = scaled_master_df.groupby('name')['weighted_score'].mean().reset_index()

    # Sort by the weighted score in descending order to find the top museums
    top_10_museums = museum_ranking.sort_values(by='weighted_score', ascending=False).head(10)

    #############################
    # Clustering using k-Means
    ##############################
    # Create a DataFrame with one row per unique museum and its aggregated scaled features
    museum_clustering_data = scaled_master_df.groupby('name').agg(
        sc_average_polarity=('sc_average_polarity', 'mean'),
        sc_museum_average_rating=('sc_museum_average_rating', 'mean'),
        sc_number_of_reviews=('sc_number_of_reviews', 'mean'),
        categories=('categories', lambda x: ', '.join(x.unique())) # Combine all unique category strings for a museum
    ).reset_index()

    # Extract all unique categories across all museums
    all_categories = museum_clustering_data['categories'].str.split(', ').explode().str.strip().unique()
    print(f"Total unique categories found: {len(all_categories)}")

    # One-hot encode the categories for each museum
    one_hot_data = {}
    for category in all_categories:
        one_hot_data[f'cat_{category}'] = museum_clustering_data['categories'].apply(lambda x: 1 if category in x else 0)

    # Concatenate all new columns at once instead of inserting one by one
    museum_clustering_data = pd.concat([museum_clustering_data, pd.DataFrame(one_hot_data)], axis=1)

    # Drop the original 'categories' column as it's now one-hot encoded
    clustering_df = museum_clustering_data.drop(columns=['name', 'categories'])

    # Determine the optimal number of clusters using the Elbow Method
    wcss = [] # Within-Cluster Sum of Squares

    # Experiment with a range of K values
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(clustering_df)
        wcss.append(kmeans.inertia_) # inertia_ is the WCSS value

    # Apply K-Means clustering with k=4
    k = 4  # Optimal number of clusters from the Elbow Method
    kmeans_final = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(clustering_df)

    # Add the cluster labels to the museum_clustering_data DataFrame
    museum_clustering_data['cluster'] = cluster_labels

    ## Analyze cluster characteristics
    museums_per_cluster = museum_clustering_data['cluster'].value_counts().sort_index()

    cluster_summary = museum_clustering_data.groupby('cluster')[['sc_average_polarity', 'sc_museum_average_rating', 'sc_number_of_reviews']].mean()

    # To analyze categories, we can look at the most frequent categories within each cluster
    cluster_dict = {}

    for i in range(k):
        cluster_i_data = museum_clustering_data[museum_clustering_data['cluster'] == i]
        all_categories_in_cluster = cluster_i_data['categories'].str.split(', ').explode().str.strip().dropna()
        cluster_dict[i] = all_categories_in_cluster.value_counts().head(10) # Top 10 categories in each cluster

    ## Visualize clusters
    # Apply PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(clustering_df)

    # Create a DataFrame for the principal components and add cluster labels
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['principal_component_1', 'principal_component_2']
    )
    pca_df['cluster'] = museum_clustering_data['cluster']

    # Create clustering plot figure (instead of showing it)
    clustering_plot = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='principal_component_1',
        y='principal_component_2',
        hue='cluster',
        data=pca_df,
        palette='viridis',
        s=100, # size of points
        alpha=0.8 # transparency
    )
    plt.title('2D PCA of Museum Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(title='Cluster')
    plt.close()  # Close to prevent display

    ########################################
    # Sentiment analysis vs. user ratings
    #########################################
    # Select relevant columns for comparison
    correlation_df = working_data[['name', 'museum_average_rating', 'average_polarity']].drop_duplicates(subset=['name'])

    # Calculate the correlation
    correlation = correlation_df['museum_average_rating'].corr(correlation_df['average_polarity'])
    print(f"Correlation between Museum Average Rating and Average Polarity: {correlation:.2f}")

    # Create correlation plot figure (instead of showing it)
    correlation_plot = plt.figure(figsize=(10, 6))
    sns.regplot(
        x='museum_average_rating',
        y='average_polarity',
        data=correlation_df,
        scatter_kws={'alpha':0.6},
        line_kws={'color':'red'}
    )
    plt.title('Correlation between Museum Average Rating and Average Review Polarity')
    plt.xlabel('Museum Average Rating')
    plt.ylabel('Average Review Polarity')
    plt.grid(True)
    plt.close()  # Close to prevent display

    ## Identifying museums with lowest correlated sentiment scores with user ratings
    # Prepare the data for regression analysis
    X = correlation_df[['museum_average_rating']]
    y = correlation_df['average_polarity']

    # Fit a simple linear regression model using scikit-learn
    model = LinearRegression()
    model.fit(X, y)

    # Calculate the residuals (actual polarity - predicted polarity)
    correlation_df['residuals'] = y - model.predict(X)

    # Identify museums with the highest absolute residuals
    # These are the museums where sentiment and rating are least aligned with the general trend
    least_correlated_museums = correlation_df.sort_values(by='residuals', key=lambda x: abs(x), ascending=False)
    
    # Return all results as a dictionary
    return {
        'working_data': working_data,
        'museum_ranking': museum_ranking,
        'top_10_museums': top_10_museums,
        'cluster_dict': cluster_dict,
        'clustering_plot': clustering_plot,
        'correlation_plot': correlation_plot,
        'least_correlated_museums': least_correlated_museums,
        'cluster_summary': cluster_summary
    }


# If this script is run directly, execute the analysis and display plots
if __name__ == "__main__":
    results = run_analysis()
    
    # Display plots
    plt.show(results['clustering_plot'])
    plt.show(results['correlation_plot'])
    
    # Print results
    print("\n=== Top 10 Museums ===")
    print(results['top_10_museums'])
    print("\n=== Cluster Summary ===")
    print(results['cluster_summary'])
    print("\n=== Least Correlated Museums ===")
    print(results['least_correlated_museums'].head(10))