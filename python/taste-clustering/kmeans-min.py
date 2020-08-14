import matplotlib.pyplot as plt
import circlify as circ
from collections import defaultdict

import os, sys, json
import seaborn as sn
import pandas as pd
import numpy as np

import spotipy
import spotipy.util as util

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

k = 6

# read data from file
feat_obj = json.load(open(os.path.join(sys.path[0], "data.txt")))
top_tracks_names = pd.DataFrame(feat_obj["names"])
top_tracks_audiofeat_df = pd.DataFrame(feat_obj["features"]).drop(
    ['analysis_url', 'track_href', 'type', 'uri', 'loudness', 'key', 'time_signature', 'mode'], axis=1)
top_tracks_df = top_tracks_names.merge(top_tracks_audiofeat_df, on="id")


# Normalize Features
def NormalizeFeatures(top_tracks_audiofeat_df):
    '''Function to normalize feature values
    top_tracks_audiofeat_df : DataFrame
    '''
    audio_features_scaler = StandardScaler()
    scaled_features = audio_features_scaler.fit_transform(
        top_tracks_audiofeat_df.drop("id", axis=1))
    scaled_features_df = pd.DataFrame(
        scaled_features, columns=top_tracks_audiofeat_df.drop('id', axis=1).columns)
    scaled_features_df["id"] = top_tracks_audiofeat_df["id"]
    return scaled_features_df


scaled_features_df = NormalizeFeatures(top_tracks_audiofeat_df)
print(scaled_features_df.head())


# PCA
pca = PCA()
principal_comp = pca.fit_transform(
    scaled_features_df.drop(["id"], axis=1))
PCA_comp = pd.DataFrame(principal_comp)


# K-Means
kmeans = KMeans(k).fit(
    scaled_features_df.drop(["id"], axis=1))
scaled_features_df["cluster"] = pd.Series(kmeans.labels_)

top_tracks_normalized_df = top_tracks_names.merge(scaled_features_df, on="id")

print("K-means results: ")
print(top_tracks_normalized_df.head())


# find top artists of each cluster
print("Calculating artist frequency...")
artist_count_df = (top_tracks_normalized_df.artist.value_counts().to_frame())
artist_count_df = artist_count_df.rename(columns={"artist": "total"})
artist_count_df['name'] = artist_count_df.index


def getArtistCountDf(k, top_tracks_normalized_df):
    count_df = top_tracks_normalized_df[top_tracks_normalized_df["cluster"]
                                        == k].artist.value_counts().to_frame()
    count_df = count_df.rename(columns={"artist": "cluster_" + str(k)})
    count_df["name"] = count_df.index
    return count_df


for i in range(k):
    artist_count_df = artist_count_df.merge(getArtistCountDf(
        i, top_tracks_normalized_df), on="name", how="outer")
artist_count_df = artist_count_df.fillna(0)

artist_genre_df = top_tracks_names.drop_duplicates("artist")[
    ["artist", "genres"]]
artist_count_df = artist_count_df.merge(
    artist_genre_df, left_on="name", right_on="artist")

print(artist_count_df.head())

# store artist counts for future use
cluster_sizes = []
for i in range(k):
    cluster_sizes.append(
        {
            "cluster": i,
            "total": artist_count_df['cluster_' + str(i)].sum()
        })

# use formula to weigh artist importance to cluster
for i in range(k):
    artist_count_df['cluster_' + str(i)] = artist_count_df['cluster_' + str(i)]**2 / (
        artist_count_df.total * artist_count_df['cluster_' + str(i)].sum())


def findBestGenre(genres):
    g = defaultdict(int)
    for row in genres:
        for genre in row:
            g[genre] += 1
    return sorted(g, key=g.get, reverse=True)


def listLimit(n, lst):
    return lst if len(lst) < n else lst[:n]


genre_groups = []
artist_groups = []
for i in range(k):
    cluster_artists_sort = artist_count_df.nlargest(
        10, columns=['cluster_' + str(i)])
    cluster_artists = cluster_artists_sort[cluster_artists_sort["cluster_" + str(
        i)] > 0]
    mygenres = findBestGenre(cluster_artists["genres"])
    genre_groups.append(mygenres)
    artist_groups.append(cluster_artists['name'].values)

# Plot cluster centroids
centers = np.array(kmeans.cluster_centers_)
principal_comp_centroids = pca.transform(
    centers)
PCA_comp_centroids = pd.DataFrame(principal_comp_centroids)


# Spotify Taste Profile Heatmap with song scatter and annotations
print("Plotting taste map...")
fig, ax = plt.subplots()
# Contour plot for clusters visualization
sn.kdeplot(PCA_comp[0], PCA_comp[1], shade=True, shade_lowest=False)
sc = plt.scatter(PCA_comp_centroids[0], PCA_comp_centroids[1],
                 c=PCA_comp_centroids.index.astype(float))


def annotateCluster(i):
    xy = (PCA_comp_centroids[0][i], PCA_comp_centroids[1][i])
    text = "{}\n{}".format(
        "/".join(listLimit(2, genre_groups[i])), "\n".join(listLimit(3, artist_groups[i])))
    ax.annotate(text, xy=xy, xytext=(20, 20), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))


annotateCluster(0)
plt.show()

# get clusters sorted by size
cluster_sizes = sorted(
    cluster_sizes, key=lambda cluster: cluster["total"], reverse=True)

# Plots packed circle chart with top genres
uniq_genres = []
set_length = 0
cluster_genres = []
size_diff = 6
circ_size = k*size_diff + size_diff

# from largest cluster to smallest cluster, find the most common genres in each cluster
for i in range(k):
    ind = cluster_sizes[i]["cluster"]
    cluster_sizes[i]["size"] = circ_size
    circ_size -= size_diff
    for genre in genre_groups[ind]:
        if genre not in uniq_genres:
            uniq_genres.append(genre)
        if len(uniq_genres) - set_length >= 3:
            break
    cluster_genres.append(uniq_genres[set_length:])
    set_length = len(uniq_genres)


def formatGenreText(genres):
    for i in range(len(genres)):
        if len(genres[i]) > 0:
            genres[i] = "\n".join(genres[i])
    return [cluster for cluster in genres if cluster != []]


genre_data = []
for i in range(k):
    if len(cluster_genres[i]) > 0:
        genre_data.append(cluster_sizes[i])

print("Plotting packed bubble chart...")
circles = circ.circlify(genre_data, datum_field="size", id_field="labels")
circ.bubbles(circles, labels=formatGenreText(cluster_genres))
