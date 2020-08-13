import os
import sys
import json
from dotenv import load_dotenv
import spotipy
import spotipy.util as util

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

top_tracks = sp.current_user_top_tracks(limit=49, time_range="long_term")
songs = top_tracks["items"]
top_tracks = sp.next(top_tracks)
songs.extend(top_tracks['items'])

# saves metadata and track ids into formatted variables
trkid = []
artistid = set()
trkname = []
for track in songs:
    trkid.append(track["id"])
    artistid.add(track["artists"][0]["id"])
    trkname.append(
        {"name": track["name"], "id": track["id"], "artist": track["artists"][0]["name"]})

# gets audio features
audiofeatures = sp.audio_features(trkid)
print(str(len(audiofeatures)) + " songs analyzed")

# gets artist genres
count = 50
artistid = list(artistid)
artists = []
while count < len(artistid):
    artists.extend(sp.artists(artistid[:count])["artists"])
    count += 50
artists.extend(sp.artists(artistid[count - 50:])["artists"])

# format artist genres into array of objects
artistgenres = []
for art in artists:
    artistgenres.append({"name": art["name"], "genres": art["genres"]})

for i in range(len(trkname)):
    for art in artistgenres:
        if trkname[i]["artist"] == art["name"]:
            trkname[i]["genres"] = art["genres"]

featObj = {
    "names": trkname,
    "features": audiofeatures
}

with open(os.path.join(sys.path[0], "data.txt"), 'w') as json_file:
    json.dump(featObj, json_file)
