import os
import hashlib
import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
from scipy.ndimage import maximum_filter
from sklearn.cluster import KMeans
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

class MusicProcessor:
    def __init__(self, audio_folder, output_folder, db_config, database_name, drop_if_exists=False):
        self.audio_folder = audio_folder
        self.output_folder = output_folder
        self.db_config = db_config
        self.database_name = database_name
        self.drop_if_exists = drop_if_exists
        self.kmeans = None
        self.cluster_to_genre = {0: 'Electronic', 1: 'Pop', 2: 'Classical'}
        self.connection = self.connect_to_db()
        self.feature_columns = [] 

    def connect_to_db(self):
        try:
            connection = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            if connection.is_connected():
                print("Connected to MySQL server")
                cursor = connection.cursor()

                # Drop the database if it exists and if drop_if_exists is set to True
                if self.drop_if_exists:
                    cursor.execute(f"DROP DATABASE IF EXISTS {self.database_name}")
                    print(f"Database '{self.database_name}' dropped successfully.")

                # Check if the database exists
                cursor.execute(f"SHOW DATABASES LIKE '{self.database_name}'")
                database_exists = cursor.fetchone()

                # Create the database if it doesn't exist
                if not database_exists:
                    cursor.execute(f"CREATE DATABASE {self.database_name}")
                    print(f"Database '{self.database_name}' created successfully.")
                else:
                    print(f"Database '{self.database_name}' already exists. Skipping creation.")

                # Reconnect to the specified database
                connection.database = self.database_name
                print(f"Connected to MySQL database '{self.database_name}'")
                return connection
        except Error as e:
            print(f"Error connecting to or creating MySQL database '{self.database_name}': {e}")
            return None

    def create_tables(self):
        if self.connection is None:
            print("No connection to MySQL. Cannot create tables.")
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    file_name VARCHAR(255),
                    fingerprint LONGTEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    file_name VARCHAR(255),
                    spectral_centroid_mean FLOAT,
                    spectral_bandwidth_mean FLOAT,
                    zero_crossing_rate_mean FLOAT,
                    tempo FLOAT,
                    average_beat_interval FLOAT,
                    std_beat_interval FLOAT,
                    num_beats INT
                )
            """)
            self.connection.commit()
            print("Tables created successfully.")
        except Error as e:
            print(f"Error creating tables: {e}")

    def generate_fingerprint(self, file_path, fan_value=10):
        y, sr = librosa.load(file_path, sr=None)
        S = np.abs(librosa.stft(y))
        local_max = maximum_filter(S, size=(20, 20)) == S
        S_max = S * local_max
        peak_coords = np.argwhere(S_max > np.percentile(S_max, 90))
        
        hashes = []
        for i in range(0, len(peak_coords), fan_value):
            for j in range(1, fan_value):
                if i + j < len(peak_coords):
                    freq1, time1 = peak_coords[i]
                    freq2, time2 = peak_coords[i + j]
                    delta_t = time2 - time1
                    if delta_t > 0:
                        hash_value = hashlib.sha1(f"{freq1}|{freq2}|{delta_t}".encode('utf-8')).hexdigest()
                        hashes.append(hash_value)
        
        return set(hashes)

    def extract_audio_features(self, file_path):
        """Extract essential audio features for clustering and classification."""
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_intervals = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
        
        features = {
            'spectral_centroid_mean': spectral_centroid,
            'spectral_bandwidth_mean': spectral_bandwidth,
            'zero_crossing_rate_mean': zero_crossing_rate,
            'tempo': float(tempo),
            'average_beat_interval': np.mean(beat_intervals) if beat_intervals.size > 0 else 0,
            'std_beat_interval': np.std(beat_intervals) if beat_intervals.size > 0 else 0,
            'num_beats': len(beat_frames)
        }
        
        # Flatten MFCC coefficients
        for i, mfcc_value in enumerate(np.mean(mfcc, axis=1)):
            features[f'mfcc_mean_{i+1}'] = mfcc_value
        
        return features


    def insert_fingerprint(self, file_name, fingerprint):
        if not fingerprint:
            fingerprint = [""]  # Fallback in case fingerprint is empty
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO fingerprints (file_name, fingerprint)
            VALUES (%s, %s)
        """, (file_name, ",".join(fingerprint)))
        self.connection.commit()

    def process_audio_files(self):
        self.create_tables()
        for file_name in os.listdir(self.audio_folder):
            file_path = os.path.join(self.audio_folder, file_name)
            if file_path.endswith(('.wav', '.mp3', '.flac')):
                if file_path.endswith('.mp3'):
                    file_path = self.convert_to_wav(file_path)
                
                try:
                    fingerprint = self.generate_fingerprint(file_path)
                    self.insert_fingerprint(file_name, fingerprint)
                    
                    features = self.extract_audio_features(file_path)
                    self.insert_features(file_name, features)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    def insert_features(self, file_name, features):
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO features (file_name, spectral_centroid_mean, spectral_bandwidth_mean,
                zero_crossing_rate_mean, tempo, average_beat_interval, std_beat_interval, num_beats)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            file_name, 
            features['spectral_centroid_mean'], 
            features['spectral_bandwidth_mean'],
            features['zero_crossing_rate_mean'], 
            features['tempo'], 
            features['average_beat_interval'],
            features['std_beat_interval'], 
            features['num_beats']
        ))
        self.connection.commit()

    def load_features(self):
        if self.connection is None:
            print("No database connection. Cannot load features.")
            return pd.DataFrame()
        
        try:
            query = "SELECT * FROM features"
            return pd.read_sql(query, self.connection)
        except Error as e:
            print(f"Error loading features: {e}")
            return pd.DataFrame()


    def convert_to_wav(self, file_path):
        output_file_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(file_path))[0] + '.wav')
        if not os.path.exists(output_file_path):
            audio = AudioSegment.from_mp3(file_path)
            audio.export(output_file_path, format="wav")
        return output_file_path

    def train_kmeans(self, n_clusters=3):
        """Train KMeans clustering model on features."""
        df = self.load_features()
        
        if df.empty:
            print("No data loaded from features table.")
            return
        
        # Exclude non-feature columns
        self.feature_columns = df.drop(columns=['id', 'file_name']).columns.tolist()
        
        input_features = df[self.feature_columns]
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = self.kmeans.fit_predict(input_features)
        df['genre'] = df['cluster'].map(self.cluster_to_genre)
        print("KMeans model trained and clusters assigned.")
        return df


    def recognize_song(self, file_path):
        new_fingerprint = self.generate_fingerprint(file_path)
        query = "SELECT file_name, fingerprint FROM fingerprints"
        fingerprints = pd.read_sql(query, self.connection)
        
        best_match, best_score = "Unknown", 0
        for _, row in fingerprints.iterrows():
            stored_fingerprint = set(row['fingerprint'].split(","))
            common_hashes = len(new_fingerprint.intersection(stored_fingerprint))
            if common_hashes > best_score:
                best_score = common_hashes
                best_match = row['file_name']
        
        return best_match

    def predict_genre(self, file_path):
        features = self.extract_audio_features(file_path)
        new_features = pd.DataFrame([features])

        # Check that self.feature_columns is available
        if not hasattr(self, 'feature_columns'):
            raise AttributeError("Feature columns not set. Please run train_kmeans first.")

        # Align new_features with feature_columns, filling missing columns with 0
        new_features = new_features.reindex(columns=self.feature_columns, fill_value=0)

        # Predict the cluster and map to genre
        cluster = self.kmeans.predict(new_features)[0]
        return self.cluster_to_genre.get(cluster, "Unknown")

    def evaluate_clustering_extended(self, df):
        """
        Evaluate clustering quality using multiple metrics.
        Requires `train_kmeans` to have been run.
        """
        if 'cluster' not in df.columns:
            print("No clustering data available. Run train_kmeans first.")
            return None

        features = df[self.feature_columns]
        labels = df['cluster']

        # Calculate evaluation metrics
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)

        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
        return silhouette, davies_bouldin, calinski_harabasz

    def plot_clustering(self, df, cluster_column='cluster', x_col='spectral_centroid_mean', y_col='spectral_bandwidth_mean'):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], c=df[cluster_column], cmap='viridis', alpha=0.6, edgecolor='k')
        plt.colorbar(label='Cluster')
        plt.xlabel("Spectral Centroid Mean")
        plt.ylabel("Spectral Bandwidth Mean")
        plt.title("Clustering of Songs Based on Audio Features")
        plt.show()

