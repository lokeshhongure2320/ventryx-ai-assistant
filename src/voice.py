import io
import speech_recognition as sr
from gtts import gTTS
import streamlit as st

def transcribe_audio(audio_bytes):
    """
    Converts audio bytes to text using Google's free Web Speech API.
    Handles common errors gracefully.
    """
    if not audio_bytes:
        return None
    
    recognizer = sr.Recognizer()
    
    # audio_recorder_streamlit returns WAV format bytes
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            # Read the entire audio file
            audio_data = recognizer.record(source)
            
        # Recognize speech using Google's API
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        st.toast("🎤 Sorry, I couldn't understand the audio. Please try speaking clearly.", icon="⚠️")
        return None
    except sr.RequestError as e:
        st.toast("🎤 Voice service is temporarily unavailable. Check your internet connection.", icon="⚠️")
        return None
    except Exception as e:
        st.toast("🎤 An error occurred while processing audio. Please try again.", icon="⚠️")
        return None

def generate_audio_response(text):
    """
    Converts text to an MP3 byte stream using Google Text-to-Speech.
    """
    if not text:
        return None
    
    try:
        # Generate TTS audio
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.toast("🔊 Could not generate voice response.", icon="⚠️")
        return None
