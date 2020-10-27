import os, sys, json, time
import pandas as pd
from dotenv import load_dotenv
import spotipy
import spotipy.util as util

data = pd.read_csv(os.path.join(sys.path[0], 'ratings.csv'))

songs = data['song'].unique()
print(len(songs))

# loads and sets env vars
load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_id = os.getenv("USERNAME")

redirect_uri = "http://localhost:9999"
scope = 'user-library-read, playlist-modify-public, playlist-modify-private user-top-read'

# obtains user token from Spotify
token = util.prompt_for_user_token(
    user_id, scope=scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)

i = 0
starttime = time.time()
df_arr = []
while i < len(songs):
  if (i % 10000 == 0):
    print(i, "done. Took", time.time() - starttime, "sec")
    token = util.prompt_for_user_token(
    user_id, scope=scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth=token)
    starttime = time.time()
    time.sleep(0.2)
  retsongs = sp.tracks(songs[i:min(i + 50, len(songs))])

  artist_in_arr = []
  track_name = []
  artist_arr = []
  for track in retsongs["tracks"]:
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
    new_row = [songs[i + ind], track_name[ind], ",".join(artist_name), ",".join(genres_list)]
    df_arr.append(new_row)
  i += 50

df_songinfo = pd.DataFrame(df_arr,columns=["id", "title", "artists", "genres"])
print(df_songinfo.head(10))

df_songinfo.to_csv('songs.csv', index=False)

print("done")


