from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def get_topics(text_list, num_clusters=5, top_n_words=5):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(text_list)

    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(X)

    terms = vectorizer.get_feature_names_out()
    clusters = {}

    for i in range(num_clusters):
        center = km.cluster_centers_[i]
        top_indices = center.argsort()[-top_n_words:][::-1]
        keywords = [terms[j] for j in top_indices]
        clusters[f"Cluster {i+1}"] = keywords

    return clusters
