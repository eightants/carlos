import os, sys
import numpy as np
import pandas as pd

from dotenv import load_dotenv
import spotipy
import spotipy.util as util

# loads and sets API keys
load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_id = os.getenv("USER_ID")

# redirect uri and scope for obtaining Spotify token
redirect_uri = "http://localhost:9999"
scope = 'user-library-read, playlist-modify-public, user-top-read user-read-private'

# obtains user token from Spotify
token = util.prompt_for_user_token(user_id, scope=scope, client_id=client_id, \
                                   client_secret=client_secret, redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth = token)

# make request for my top tracks
top_tracks = sp.current_user_top_tracks(limit=49, time_range="long_term")
songs = top_tracks["items"]
top_tracks = sp.next(top_tracks) # there are 2 pages of top tracks
songs.extend(top_tracks['items'])

i = 0
df_arr = []
while i < len(songs):
  retsongs = songs[i:min(i + 50, len(songs))]

  artist_in_arr = []
  track_name = []
  artist_arr = []
  for track in retsongs:
    track_name.append(track["name"])
    a_tmp = []
    for artist in track["artists"]:
      artist_arr.append(artist["id"])
      a_tmp.append(artist["id"])
    artist_in_arr.append(a_tmp)
  artists_api = list(set(artist_arr))
  j = 0
  artist_info = []
  while j < len(artists_api):
    ret_art = sp.artists(artists_api[j:min(j + 50, len(artists_api))])
    artist_info += ret_art["artists"]
    j += 50
  genres = {}
  for artist in artist_info:
    genres[artist["id"]] = {'genres': artist["genres"], 'name': artist["name"]}
  for ind in range(len(artist_in_arr)):
    item = artist_in_arr[ind]
    artist_name = []
    genres_list = []
    for artist in item:
      artist_name.append(genres[artist]["name"])
      genres_list += genres[artist]["genres"]
    new_row = [songs[i + ind]["id"], track_name[ind], ",".join(artist_name), ",".join(genres_list)]
    df_arr.append(new_row)
  i += 50

my_songinfo = pd.DataFrame(df_arr,columns=["id", "title", "artists", "genres"])
my_songinfo.head()

rating_data = []
my_user = sp.current_user()
counter = 0
total = len(songs) // 5 + 1

for song in songs:
    rating_data.append([my_user["id"], song["id"], 5 - counter//total, my_user["country"]])
    counter += 1

my_ratinginfo = pd.DataFrame(rating_data,columns=["user", "song", "rating", "country"])
my_ratinginfo.head()

my_ratinginfo.to_csv(os.path.join(sys.path[0], '_myratings.txt'), index=False)
my_songinfo.to_csv(os.path.join(sys.path[0], '_mysongs.txt'), index=False)

