import streamlit as st
import pandas as pd
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn import preprocessing
import csv

# Fungsi untuk mengambil token dari Spotify API
def getToken():
    client_id = st.text_input("Enter Client ID")
    client_secret = st.text_input("Enter Client Secret")
    auth_url = "https://accounts.spotify.com/api/token"
    auth_header = {
        'Authorization': 'Basic ' + (client_id + ":" + client_secret).encode('ascii').decode('utf-8')}
    auth_data = {
        'grant_type': 'client_credentials'
    }
    auth_response = requests.post(auth_url, headers=auth_header, data=auth_data)
    if auth_response.status_code == 200:
        access_token = auth_response.json()['access_token']
        return access_token
    else:
        st.write("Error in fetching token")
        return None

# Fungsi untuk membuat header autentikasi dengan token
def getAuthHeader(token):
    return {
        'Authorization': 'Bearer ' + token
    }

# Fungsi untuk mengambil fitur audio dari track
def getAudioFeatures(token, trackId):
    url = f"https://api.spotify.com/v1/audio-features/5l3jhWIfRg1FeKgw7R1jWb"
    headers = getAuthHeader(token)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        audio_features = response.json()
        return audio_features
    else:
        st.write(f"Error fetching audio features for track {trackId}: {response.status_code}")
        return None

# Fungsi untuk mengambil lagu dari playlist
def getPlaylistItems(token, playlistId):
    url = f'https://api.spotify.com/v1/playlists/5PfGjvJ1aw3EyDfvq59IKG/tracks'
    limit = '&limit=100'  
    market = '?market=ID'  
    fields = '&fields=items%28track%28id%2Cname%2Cartists%2Cpopularity%2C+duration_ms%2C+album%28release_date%29%29%29'
    url = url + market + fields + limit  
    headers = getAuthHeader(token)
    result = requests.get(url, headers=headers)
    
    if result.status_code == 200:
        json_result = result.json()
        dataset = []
        dataset2 = []
        
        for item in json_result['items']:
            track = item['track']
            playlist_items_temp = [
                track['id'],
                track['name'],
                track['artists'][0]['name'],
                track['popularity'],
                track['duration_ms'],
                int(track['album']['release_date'][:4])
            ]
            dataset.append(playlist_items_temp)

            audio_features = getAudioFeatures(token, track['id'])
            if audio_features:
                dataset2.append([audio_features.get(key, None) for key in 
                                 ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])

        return dataset, dataset2
    else:
        st.write("Error fetching playlist items")
        return None, None

# Fungsi untuk melakukan preprocessing dan clustering data
def dataProcessing():
    data = pd.read_csv('dataset.csv')
    st.write("## Preprocessing Result")
    
    data = data[['artist', 'name', 'year', 'popularity', 'key', 'mode', 'duration_ms', 
                 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 
                 'liveness', 'speechiness', 'tempo', 'valence']]
    data = data.drop(['mode'], axis=1)
    data['artist'] = data['artist'].map(lambda x: str(x)[2:-1])
    data['name'] = data['name'].map(lambda x: str(x)[2:-1])
    data = data[data['name'] != '']
    st.write("### Data after cleaning:")
    st.write(data)

    st.write("## Normalization Result")
    data2 = data.copy()
    data2 = data2.drop(['artist', 'name', 'year', 'popularity', 'key', 'duration_ms'], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data2)
    data2 = pd.DataFrame(x_scaled)
    data2.columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                     'loudness', 'liveness', 'speechiness', 'tempo', 'valence']
    
    st.write(data2)

    st.write("## Dimensionality Reduction with PCA")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data2)
    pca_df = pd.DataFrame(data=pca_data, columns=['x', 'y'])
    fig = px.scatter(pca_df, x='x', y='y', title='PCA')
    st.plotly_chart(fig)

    st.write("## Clustering with K-Means")
    data2 = list(zip(pca_df['x'], pca_df['y']))
    kmeans = KMeans(n_init=10, max_iter=1000).fit(data2)
    fig = px.scatter(pca_df, x='x', y='y', color=kmeans.labels_,
                     color_continuous_scale='rainbow', hover_data=[data.artist, data.name])
    st.plotly_chart(fig)

    st.write("Enjoy!")

# Fungsi utama untuk menampilkan streamlit
def main():
    st.write("# Spotify Playlist Clustering")
    
    client_id = st.text_input("Enter Client ID")
    client_secret = st.text_input("Enter Client Secret")
    playlist_id = st.text_input("Enter Playlist ID")

    if st.button('Create Dataset!'):
        token = getToken()
        if token:
            dataset, dataset2 = getPlaylistItems(token, playlist_id)
            if dataset and dataset2:
                with open('dataset.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["id", "name", "artist", "popularity", "duration_ms", "year", 
                                     "danceability", "energy", "key", "loudness", "mode", 
                                     "speechiness", "acousticness", "instrumentalness", 
                                     "liveness", "valence", "tempo"])
                    for i in range(len(dataset)):
                        writer.writerow(dataset[i] + dataset2[i])
                st.write("Dataset created successfully!")

                # Call dataProcessing after dataset is created
                dataProcessing()

if __name__ == "__main__":
    main()
