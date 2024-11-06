import speech_recognition as sr
from pydub import AudioSegment

def basic_speech_to_text(audio_path):
    # Convert audio to WAV format for compatibility with pocketsphinx
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)  # Convert to mono for better recognition accuracy
    audio.export("temp.wav", format="wav")

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)

    # Transcribe audio to text using pocketsphinx (offline)
    try:
        text_output = recognizer.recognize_sphinx(audio_data)
        print("Transcribed Text:\n")
        print(text_output)
    except sr.UnknownValueError:
        print("No understandable speech found in the audio.")
    except sr.RequestError:
        print("Sphinx engine is not available.")

# Replace 'your_audio_file.mp3' with the path to your audio file
basic_speech_to_text(r'C:\Users\admin\Downloads\Flyers (OPOpening).mp3')
