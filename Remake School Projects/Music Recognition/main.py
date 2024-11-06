from music_processor import MusicProcessor

# MySQL Database Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1',
    'database': 'musicdb'
}

# Initialize the processor
processor = MusicProcessor(
    audio_folder=r'music', 
    output_folder=r'music_wav', 
    db_config=db_config,
    database_name= db_config['database'],
    drop_if_exists=True
)

# Drop and recreate the database
# processor.drop_database('test')

# Process audio files and save to MySQL
processor.process_audio_files()

# Train the KMeans model

# Load features and plot clusters
df = processor.train_kmeans()
# if not df.empty:
processor.plot_clustering(df)

# Predict the genre for a new audio file
predicted_genre = processor.predict_genre(r"C:\Users\admin\Downloads\Death Parade - Opening _ Flyers [ ezmp3.cc ].mp3")
print(f"Predicted genre: {predicted_genre}")

# Recognize a song by matching fingerprints
recognized_song = processor.recognize_song(r"C:\Users\admin\Downloads\Death Parade - Opening _ Flyers [ ezmp3.cc ].mp3")
print(f"Recognized song: {recognized_song}")
