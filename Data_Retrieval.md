---
title: store fake playlists
notebook: Data_Retrieval.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import os
import pickle
import h5py
import json
%matplotlib inline
```





```python
'''
Initialize Spotify API connection. Note: we've removed our client ID / secret from this public repository;
please use your own credentials should you wish to run this notebook.
'''

def refresh():
    os.environ["SPOTIPY_CLIENT_ID"] = 'XXX'
    os.environ["SPOTIPY_CLIENT_SECRET"] = 'XXX'
    os.environ["SPOTIPY_REDIRECT_URI"] = 'http://localhost/'

    scope = 'user-library-read'

    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        #print("Usage: %s username" % (sys.argv[0],))
        sys.exit()

    token = util.prompt_for_user_token(username, scope)

    # Print some tracks
    if token:
        sp = spotipy.Spotify(auth=token)
        results = sp.current_user_saved_tracks()
        for item in results['items']:
            track = item['track']
            #print(track['name'] + ' - ' + track['artists'][0]['name'])
    else:
        print("Can't get token for", username)
    return sp

sp = refresh()
```




```python
categories = sp.categories(country=None, locale=None, limit=50, offset=0)

#Get the ids of categories
temp = categories['categories']

cat_ids = []
for item in temp['items']:
    cat_ids.append(item['id'])

```




```python

playlists = {}
for cat_id in cat_ids:
    playlists[cat_id] = sp.category_playlists(category_id=cat_id, country=None, limit=50, offset=0)

playlist_ids_by_cat = {}
for category, playlist in playlists.items():
    #print(playlist['playlists']['items'][0]['id'])
    playlist_ids_by_cat[category] = [x['id'] for x in playlist['playlists']['items']]

```




```python
'''
Useful track-getting functions!
'''

def spotify_id_to_isrc(spotify_ids):
    '''
    converts spotify ids to isrcs
    '''
    tracks = sp.tracks(spotify_ids)
    return [x['external_ids']['isrc']  for x in tracks['tracks']]

def isrc_to_spotify_id(isrcs):
    '''
    converts isrcs to spotify ids
    This takes a while since we need to search
    Note: isrc --> spotify_id is not necessarily a one-to-one mapping (multiple spotify ids
    can map to the same isrc)
    ''' 
    ids = []
    for isrc in isrcs:
        ids.append(sp.search('isrc:'+isrc)['tracks']['items'][0]['id'])
    return ids

def get_popularity_and_markets(spotify_ids):
    '''
    Given a list of spotify IDs, get the popularity and market info for the songs
    '''
    chunk_size= 42
    tmp = {}
    for i in range(0, len(spotify_ids), chunk_size):
        chunk = spotify_ids[i:i+chunk_size]
        features = sp.tracks(chunk)['tracks']
        features  = pd.DataFrame([x for x in features if isinstance(x, dict)])
        tmp_df = pd.DataFrame(features)
        tmp_df.index = tmp_df['id']
        tmp_df = tmp_df[['id', 'popularity', 'available_markets']]
        tmp_df['available_markets'] = tmp_df['available_markets'].apply(len)
        tmp.update(tmp_df.T.to_dict())
    df = pd.DataFrame(tmp).T
    return df

def get_followers(playlist_id, user = 'spotify'):
    '''
    Get the # of follower of a playlist
    '''
    playlist = sp.user_playlist(user, playlist_id=playlist_id, fields = ['followers'])
    return playlist['followers']['total']

#Default: US, 11/24/2017 at 8PM
def get_featured_playlists(country = 'US', time = '2017-11-24T18:00:00'):
    '''
    Get whether or not a playlist was featured on a certain date at a certain time.
    '''
    featured = sp.featured_playlists(country=country, timestamp=time, limit=50, offset=0)
    return [x['id'] for x in featured['playlists']['items']]

def get_track_ids(playlist_id = '37i9dQZF1DX3FNkD0kDpDV'):
    ''' 
    Given a Spotify Playlist ID, returns a list of spotify ids for songs in playlist
    '''
    offset = 0
    playlist = sp.user_playlist_tracks(user = 'spotify', playlist_id = playlist_id, limit = 100)
    ids = [x['track']['id']  for x in playlist['items']]
    # if we hit the limit, need to add more
    while len(ids) / (offset + 100) == 1:
        offset = offset + 100
        playlist = sp.user_playlist_tracks(user = 'spotify', playlist_id = playlist_id, limit = 100, offset = offset)
        ids = ids + [x['track']['id']  for x in playlist['items']]
    return ids

def get_track_audio_features(spotify_ids = get_track_ids()):
    'Given a list of spotify IDs, returns a dataframe of track audio features'
    chunk_size= 42
    tmp = {}
    for i in range(0, len(spotify_ids), chunk_size):
        chunk = spotify_ids[i:i+chunk_size]
        features = sp.audio_features(chunk)
        if features:
            tmp_df = pd.DataFrame([x for x in features if isinstance(x, dict)])
            tmp_df.index = tmp_df['id']
            tmp.update(tmp_df.T.to_dict())
    df = pd.DataFrame(tmp).T
    df = df.drop(['analysis_url', 'track_href', 'uri', 'type'], 1)
    return df

```




```python
import time
def get_playlist_data(playlist_ids):
    '''
    Given a list of Spotify playlist IDs, returns a dataframe containing a row
    for each inputed playlist with columns for the following data:
    1) *average* audio characteristics for the songs in that playlist:
        acousticness, danceability, duration,
        energy, instrumentalness, key, liveness, loudness, mode, tempo,
        valence, and time signature
    2) average popularity of songs in the playlist
    3) popularity of most popular song in playlist (might be an anchor song to the playlist)
    4) average # of markets the songs in the playlist are available i
    5) global playlist info
        - number of followers the playlist has (response variable?)
        - number of tracks in playlist
        - whether or not the playlist was "featured" on 11/24/2017 at 8PM
    '''
    rez = {}
    # force list
    if not isinstance(playlist_ids, list):
        playlist_ids = [playlist_ids]
        
    featured_playlists = get_featured_playlists()
    audio_char_dict = {}
    popularity_dict = {}
    for playlist_id in playlist_ids:
        print('Getting info for: ' + playlist_id)
        tmp = {}
        try:
            track_ids = get_track_ids(playlist_id)
        except spotipy.client.SpotifyException:
            print('WARNING: Playlist does not exist. Skipping.')
            continue
        except:
            time.sleep(10)
            track_ids = get_track_ids(playlist_id)
        # get average audio characteristics
        audio_chars = get_track_audio_features(track_ids)
        audio_char_mean = audio_chars.mean().to_dict()
        # get popularity and markets
        pop_and_mkts = get_popularity_and_markets(track_ids)
        pop_and_mkts_mean = pop_and_mkts.mean().to_dict()
        audio_char_dict.update(audio_chars.T.to_dict())
        popularity_dict.update(pop_and_mkts.T.to_dict())
        # get # followers
        tmp['num_followers'] = get_followers(playlist_id)
        tmp['num_tracks'] = len(track_ids)
        tmp['featured'] = 1 if playlist_id in featured_playlists else 0
        tmp.update(audio_char_mean)
        tmp.update(pop_and_mkts_mean)
        rez[playlist_id] = tmp
    return pd.DataFrame(rez).T, popularity_dict, audio_char_dict


progress = 0
playlist_level_data = {}
for cat, playlists in playlist_ids_by_cat.items():
    if playlists[0] in playlist_level_data.keys():
        continue
    sp = refresh()
    #print('Starting Category: ' + cat)
    playlist_data, pop, audio = get_playlist_data(playlists)
    playlist_data['category'] = cat
    song_level_data_pop.update(pop)
    song_level_data_audio.update(audio)
    playlist_level_data.update(playlist_data.T.to_dict())



```




```python
'''
Store all of this data. Uncomment + re-run ONLY IF YOU WANT TO OVERWRITE ALL OF THE CURRENT DATA
'''

```




```python
'''
Loop through all of our playlists and store the track IDs from each playlists;
we use this to get a mapping of playlists to tracks (could theoreticallly have done this above,
but we tragically did not.)
'''
playlist_to_tracks = {}
progress = 1
for cat, playlists in playlist_ids_by_cat.items():
    print(progress / len(playlist_ids_by_cat))
    for playlist in playlists:
        try:
            track_ids = get_track_ids(playlist)
        except spotipy.client.SpotifyException:
            print('WARNING: Playlist does not exist. Skipping.')
            continue
        except:
            time.sleep(10)
            track_ids = get_track_ids(playlist)
        playlist_to_tracks[playlist] = track_ids
    progress += 1
  
```




```python

#pickle.dump(playlist_to_tracks, open( "playlist_to_track_20171203.p", "wb" ) )
```





```python
'''
Merge in Million Songs Database
'''

'''
Get links from Echonest to match MSD to Spotify data
'''

spotify_ids_in_playlists = set(song_level_data_pop.index).union(set(song_level_data_audio.index))
loc = 'millionsongdataset_echonest.tar/millionsongdataset_echonest/millionsongdataset_echonest/'

def msd_to_spotify(msd_id):
    folder = msd_id[2:4]
    with open(loc + folder + '/' + msd_id+'.json') as f:
        file = json.load(f)
        try:
            tmp = file['response']['songs'][0]['tracks']
            spotify_ids = [x['foreign_id'] for x in tmp if 'spotify' in x['catalog']]
            ids = [y.split('spotify:track:')[1] for y in spotify_ids]
            ids = [x for x in ids if x in spotify_ids_in_playlists]
        except:
            return np.NaN
    if ids:
        return ids[0]
    else:
        return np.NaN
    
msd_id = pd.Series(msd_data.index, index = msd_data.index)
msd_to_spotify_map = msd_id.apply(msd_to_spotify)

with open('msd_map_20171203.p', 'wb') as f:
    pickle.dump(msd_to_spotify_map, f)
    
with open('msd_map_20171203.p', 'rb') as f:
    msd_map = pickle.load(f).dropna()
    msd_map = pd.Series(msd_map.to_dict())

msd_summary = h5py.File('msd_summary_file.h5', 'r')
idx = msd_summary['metadata']['songs']['song_id']
idx = [x.decode('ascii') for x in idx]
song_hotness = msd_summary['metadata']['songs']['song_hotttnesss']
artist_hotness = msd_summary['metadata']['songs']['artist_hotttnesss']
artist_familiarity = msd_summary['metadata']['songs']['artist_familiarity']

msd_data = pd.DataFrame([(x,y,z) for x,y,z in zip(song_hotness, artist_hotness, artist_familiarity)], index = idx)
msd_data = msd_data.loc[msd_map.index]
msd_data = pd.DataFrame(msd_data.values, index = [msd_map[x] for x in msd_data.index])
msd_data.columns = ['song_hot', 'artist_hot', 'artist_famil']

with open('msd_data_20171203.p', 'wb') as f:
    pickle.dump(msd_data, f)

```




```python

rez = {}
for playlist, tracks in playlist_to_track.items():
    audio_data = song_level_data_audio.reindex(tracks)
    popularity_data = song_level_data_pop.reindex(tracks)
    operations = {'max':np.max, 'mean':np.mean, 'median':np.median, 'min':np.min, 'sd':np.std}
    combined = audio_data.join(popularity_data, rsuffix ='_pop').join(msd_data)
    columns = [x for x in combined.columns if x not in ['id', 'id_pop']]
    rez[playlist] = {}
    for operation_name, operation in operations.items():
        new = combined[columns].apply(operation)
        new.index = [x+'_'+operation_name for x in columns]
        new.name = operation_name
        rez[playlist].update(new)
    for feature in ['key', 'mode', 'time_signature']:
        rez[playlist][feature+'_'+'mode'] = combined[feature].mode().values[0]

def aggregate_to_playlist_level(tracks):
    rez = {}
    audio_data = song_level_data_audio.reindex(tracks)
    popularity_data = song_level_data_pop.reindex(tracks)
    operations = {'max':np.max, 'mean':np.mean, 'median':np.median, 'min':np.min, 'sd':np.std}
    combined = audio_data.join(popularity_data, rsuffix ='_pop').join(msd_data)
    columns = [x for x in combined.columns if x not in ['id', 'id_pop']]
    for operation_name, operation in operations.items():
        new = combined[columns].apply(operation)
        new.index = [x+'_'+operation_name for x in columns]
        new.name = operation_name
        rez.update(new.to_dict())
    for feature in ['key', 'mode', 'time_signature']:
        rez[feature+'_'+'mode'] = combined[feature].mode().values[0]
    rez['category'] = track_categories.reindex(tracks).mode()[0]
    rez['num_tracks'] = len(tracks)
    rez['featured'] = 0
    return rez

aggregate_data = pd.DataFrame(rez).T
aggregate_data = aggregate_data.join(playlist_level_data[['category', 'featured', 'num_followers', 'num_tracks']])
aggregate_data = aggregate_data.dropna(subset = ['category'])

vars_to_keep = ['acousticness_max', 'acousticness_mean', 'acousticness_median', 'acousticness_min', 'acousticness_sd', 
'category', 'danceability_max', 'danceability_mean', 'danceability_median', 'danceability_min', 
'danceability_sd', 'duration_ms_max', 'duration_ms_mean', 'duration_ms_median', 'duration_ms_min', 
'duration_ms_sd', 'energy_max', 'energy_mean', 'energy_median', 'energy_min', 'energy_sd', 'featured',
'instrumentalness_max', 'instrumentalness_mean', 'instrumentalness_median', 'instrumentalness_min', 
'instrumentalness_sd', 'key_mode', 'liveness_max', 'liveness_mean', 'liveness_median', 'liveness_min', 
'liveness_sd', 'loudness_max', 'loudness_mean', 'loudness_median', 'loudness_min', 'loudness_sd', 
'mode_mean', 'mode_median', 'mode_mode', 'available_markets_max', 'available_markets_mean', 
           'available_markets_median', 'available_markets_min', 'available_markets_sd', 'popularity_max', 
'popularity_mean', 'popularity_median', 'popularity_min', 'popularity_sd', 'speechiness_max', 
'speechiness_mean', 'speechiness_median', 'speechiness_min', 'speechiness_sd', 'tempo_max', 'tempo_mean', 
'tempo_median', 'tempo_min', 'tempo_sd', 'time_signature_mode', 'valence_max', 'valence_mean', 
'valence_median', 'valence_min', 'valence_sd', 'artist_famil_max', 'artist_famil_mean', 
'artist_famil_median', 'artist_famil_min', 'artist_famil_sd', 'artist_hot_max', 
'artist_hot_mean', 'artist_hot_median', 'artist_hot_min', 'artist_hot_sd', 
'song_hot_max', 'song_hot_mean', 'song_hot_median', 'song_hot_min', 
'song_hot_sd', 'num_followers'
          , 'num_tracks'
          ]

with open('aggregate_data.p', 'wb') as f:
    data = pickle.dump(aggregate_data[vars_to_keep], f)
```




```python
'''
Determine track-level category based on playlists that contain it -- this is a rather crude measure, but works well
'''

track_category = {}
for playlist, tracks in playlist_to_track.items():
    try:
        playlist_category = playlist_level_data['category'][playlist]
    except:
        pass
    for track in tracks:
        if track_category.get(track, None):
            if track_category[track].get(playlist_category, None):
                track_category[track][playlist_category] += 1
            else:
                track_category[track][playlist_category] = 1
        else:
            track_category[track] = {playlist_category : 1}
rez = {}
for track, d in track_category.items():
    rez[track] = pd.Series(d).idxmax()
    
#Store this
with open('track_categories_20171203.p', 'wb') as f:
    pickle.dump(rez, f)
```




```python

with open('track_categories_20171203.p', 'rb') as f:
    track_categories = pd.Series(pickle.load(f))

num_per_category = 100
candidate_playlists = {}
playlist_id_to_songs = {}
possible_songs = list(spotify_ids_in_playlists)
possible_categories = list(set(track_categories.values))
i = 0
for category in possible_categories:
    songs_in_cat = track_categories.where(track_categories == category).dropna().index
    for j in range(0, num_per_category):
        tracks = np.random.choice(songs_in_cat, 30, replace = False)
        candidate_playlists[i] = aggregate_to_playlist_level(tracks)
        playlist_id_to_songs[i] = tracks
        i += 1
candidate_playlists = pd.DataFrame(candidate_playlists).T
```




```python
with open('candidate_playlists_to_songs_20171205.p', 'wb') as f:
    pickle.dump(playlist_id_to_songs, f)
with open('candidate_playlists_20171205.p', 'wb') as f:
    pickle.dump(candidate_playlists[[x for x in vars_to_keep if x != 'num_followers']], f)
```

