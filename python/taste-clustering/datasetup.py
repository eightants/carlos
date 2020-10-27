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
user_id = os.getenv("USER_ID")

redirect_uri = "http://localhost:9999"
scope = 'user-library-read, playlist-modify-public, playlist-modify-private user-top-read'

# obtains user token from Spotify
token = util.prompt_for_user_token(
    user_id, scope=scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)

print("Enter the type of data to use: ")
print("1. My Top Songs\n2. My Saved Songs\n3. A Playlist")
choice = input("Choice: ")

top_tracks = []
if choice == "1":
    top_tracks = sp.current_user_top_tracks(limit=49, time_range="long_term")
elif choice == "2":
    top_tracks = sp.current_user_saved_tracks(limit=50)
elif choice == "3":
    playlist_id = input("Enter playlist id: ")
    top_tracks = sp.playlist_tracks(playlist_id)
else:
    print("Invalid choice")
    exit(0)

# get all the songs
songs = top_tracks["items"]
while top_tracks['next']:
    top_tracks = sp.next(top_tracks)
    songs.extend(top_tracks['items'])

if choice != "1":
    for i in range(len(songs)):
        songs[i] = songs[i]["track"]


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
count = 0
audiofeatures = []
while count + 100 < len(trkid):
    audiofeatures.extend(sp.audio_features(trkid[count:count + 100]))
    count += 100
audiofeatures.extend(sp.audio_features(trkid[count:]))
print(str(len(audiofeatures)) + " songs analyzed")

# gets artist genres
count = 0
artistid = list(artistid)
artists = []
while count + 50 < len(artistid):
    artists.extend(sp.artists(artistid[count:count + 50])["artists"])
    count += 50
artists.extend(sp.artists(artistid[count:])["artists"])

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
