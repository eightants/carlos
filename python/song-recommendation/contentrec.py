from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import numpy as np
import pandas as pd

is_personalised = input("Do you want to run this program with your personal listening data? (y/n): ")
personalised = (is_personalised == "y")

data = pd.read_csv(os.path.join(sys.path[0], '_rating.csv'))
df_songs = pd.read_csv(os.path.join(sys.path[0], 'songs.csv'))

username = "1"
if personalised:
    # Add personal song data onto data from CSV
    my_ratinginfo = pd.read_csv(os.path.join(sys.path[0], '_myratings.txt'))
    my_songinfo = pd.read_csv(os.path.join(sys.path[0], '_mysongs.txt'))
    username = my_ratinginfo.loc[0,"user"]
    data = pd.concat([data,my_ratinginfo])
    df_songs = pd.concat([df_songs,my_songinfo]).drop_duplicates(subset='id', keep="first")
    print(data.shape, df_songs.shape)
else:
    username = input("Enter user index to recommend (Between 1 and " + len(data) + "): ")
print(username)


# preprocessing for string recognition, get data into metadata column, full song title
df_songs['metadata'] = df_songs["genres"].apply(
    lambda x: str(x).replace(",", " "))
# df_songs['metadata'] = df_songs["artists"].apply(lambda x: x.replace(" ", "")) + " " + df_songs["genres"].apply(lambda x: str(x).replace(",", " "))
df_songs['full'] = df_songs["title"] + " - " + df_songs["artists"]
df_metadata = df_songs[["id", "full", "metadata"]]

df_joined = data.merge(df_songs, left_on="song", right_on="id")
print(df_joined.shape)

print(df_joined[df_joined.song == "0nbXyq5TXYPCO7pr3N8S4I"].head())
df_joined.fillna("", inplace=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_metadata['metadata'])
print(tfidf_matrix.shape)
  

def get_content_rec(user):
    user_songs = df_joined[df_joined.user == user]
    user_sims = []
    for index, row in user_songs.iterrows():
        new_song_vector = tfidf.transform([row["metadata"]])
        user_sims.append(linear_kernel(new_song_vector, tfidf_matrix))
    user_scores = sum(user_sims)[0] / len(user_songs)
    sim_scores = []
    for i in range(len(user_scores)):
        sim_scores.append(
            [i, user_scores[i], df_metadata.iloc[i, 0], df_metadata.iloc[i, 1]])
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores


song_grouped = data.groupby(['song']).agg({'rating': 'count'}).reset_index()
grouped_sum = song_grouped['rating'].sum()
song_grouped['percent'] = song_grouped['rating'].div(grouped_sum)*100
song_grouped = song_grouped.sort_values(['rating', 'song'], ascending=[0, 1])

users = data['user'].unique()


reader = Reader(rating_scale=(1, 5))
rating_data = Dataset.load_from_df(data[['user', 'song', 'rating']], reader)

trainset, testset = train_test_split(rating_data, test_size=.25)
algorithm = SVD(n_epochs=5, verbose=True)
algorithm.fit(trainset)
predictions = algorithm.test(testset)

accuracy.rmse(predictions)


def pred_user_rating(ui):
    if ui in data.user.unique():
        ui_list = data[data.user == ui].song.tolist()
        predictedL = []
        songs_to_check = list(set(data.song.tolist()))
        for i in songs_to_check:
            if i not in ui_list:
                predicted = algorithm.predict(ui, i)
                predictedL.append((i, predicted[3] / 5))
        top_predictions = sorted(predictedL, key=lambda x: x[1], reverse=True)
        return top_predictions


content_df = pd.DataFrame(get_content_rec(username), columns=[
                          "idx", "sim_score", "song", "full"])
collab_df = pd.DataFrame(pred_user_rating(
    username), columns=["song", "col_score"])
df_hybrid = content_df.merge(collab_df, on="song")
df_hybrid.fillna(0, inplace=True)
df_hybrid["hybrid"] = df_hybrid["sim_score"] + df_hybrid["col_score"]
df_hybrid = df_hybrid.sort_values(
    "hybrid", ascending=False).drop_duplicates(subset=['full'])
print(df_hybrid.head(20))

df_hybrid.head(30).to_csv(os.path.join(sys.path[0], username + '_recsongs.txt'), index=False)
