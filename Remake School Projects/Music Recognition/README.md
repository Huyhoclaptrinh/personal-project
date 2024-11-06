
# Music Recognition Project

This Music Recognition project identifies songs and predicts genres using audio processing and machine learning. The project includes audio feature extraction, song fingerprinting, and genre classification with KMeans clustering, all stored in a MySQL database.

## Project Structure

- **main.py**: Main script to run the music processing and recognition workflow.
- **music_processor.py**: Contains the `MusicProcessor` class, which handles audio processing, feature extraction, MySQL storage, and genre prediction.

## Features

1. **Audio Feature Extraction**: Extracts features such as spectral centroid, bandwidth, zero-crossing rate, and beat intervals using `librosa`.
2. **Song Fingerprinting**: Generates unique fingerprints for audio files for song recognition.
3. **Genre Prediction**: Trains a KMeans clustering model on audio features and predicts the genre.
4. **Database Integration**: Stores fingerprints and audio features in a MySQL database for efficient retrieval.
5. **Song Recognition**: Matches a song's fingerprint with stored fingerprints to identify songs.

## Setup and Requirements

1. **Python Packages**: Install the following packages:
   ```bash
   pip install numpy pandas librosa pydub mysql-connector-python matplotlib
   ```

2. **MySQL Setup**: Configure the MySQL database in `main.py`:
   - Host: `localhost`
   - User: `root`
   - Password: `your_password`
   - Database: `musicdb`

3. **Folder Structure**:
   - Create an `audio_folder` to store input music files.
   - Set up an `output_folder` for WAV files after conversion.

## Usage

1. **Run Main Script**: Run `main.py` to initialize the `MusicProcessor`, process audio files, and save features and fingerprints to the MySQL database:
   ```bash
   python main.py
   ```

2. **Train KMeans Model**: The script trains a KMeans clustering model and plots clusters based on extracted audio features.

3. **Predict Genre**: Use `predict_genre` with a new audio file to classify its genre.

4. **Recognize Song**: Use `recognize_song` to identify a song by matching its fingerprint with stored records.

## Example Commands

```python
# Process audio files and save to MySQL
processor.process_audio_files()

# Predict genre for a new audio file
predicted_genre = processor.predict_genre('path/to/audio.mp3')
print(f"Predicted Genre: {predicted_genre}")

# Recognize a song by fingerprint
recognized_song = processor.recognize_song('path/to/audio.mp3')
print(f"Recognized Song: {recognized_song}")
```

## Notes

- Ensure that `.gitignore` includes `__pycache__/` to avoid tracking compiled Python files.
- Adjust the `cluster_to_genre` mapping in `MusicProcessor` as needed to suit your genre classification needs.

## License

This project is provided as-is for educational purposes.
