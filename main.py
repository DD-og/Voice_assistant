import streamlit as st
from gtts import gTTS
import sounddevice as sd
import numpy as np
import io
from pydub import AudioSegment
from pydub.playback import play
import wave
import tempfile
import os
import speech_recognition as sr
from deep_translator import GoogleTranslator
import time
import groq

GROQ_API_KEY = "your-api-key-here" 

# Set up Groq client
client = groq.Groq(api_key=GROQ_API_KEY)

@st.cache_data
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def play_audio(audio_fp, speed=1.3):
    audio = AudioSegment.from_file(audio_fp, format="mp3")
    faster_audio = audio.speedup(playback_speed=speed)
    play(faster_audio)

def record_audio(duration=5, sample_rate=16000):
    st.write("Recording... Speak now.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(duration / 100)
        progress_bar.progress(i + 1)
    
    sd.wait()
    return audio_data.flatten()

def save_audio(audio_data, sample_rate=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        with wave.open(temp_audio.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return temp_audio.name

@st.cache_data
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error connecting to the speech recognition service."

@st.cache_data
def translate_text(text, target_lang='en'):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

@st.cache_data
def process_command(command):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond to the user's command or question."
            },
            {
                "role": "user",
                "content": command
            }
        ],
        model="llama-3.1-8b-instant",
        max_tokens=1000
    )
    return chat_completion.choices[0].message.content

def main():
    st.title("Voice Assistant Web App (Llama 3.1 8B)")

    languages = {
        'English': 'en', 
        'French': 'fr', 
        'Spanish': 'es', 
        'German': 'de', 
        'Italian': 'it', 
        'Japanese': 'ja',
        'Hindi': 'hi',
        'Gujarati': 'gu'
    }

    col1, col2 = st.columns(2)
    with col1:
        input_lang = st.selectbox("Input language", list(languages.keys()))
    with col2:
        output_lang = st.selectbox("Output language", list(languages.keys()))

    input_method = st.radio("Choose input method", ["Voice", "Text"])

    speech_speed = st.slider("Speech Speed", min_value=1.0, max_value=2.0, value=1.3, step=0.1)

    if input_method == "Voice":
        if st.button("Start Listening"):
            audio_data = record_audio(duration=5)
            audio_file = save_audio(audio_data)
            command = transcribe_audio(audio_file)
            os.unlink(audio_file)
            
            st.write(f"You said: {command}")
            
            if command and command != "Sorry, I couldn't understand the audio.":
                if input_lang != 'English':
                    command = translate_text(command, 'en')
                
                response = process_command(command)
                
                if output_lang != 'English':
                    response = translate_text(response, languages[output_lang])
                
                st.write("Assistant:", response)
                audio_fp = text_to_speech(response, languages[output_lang])
                play_audio(audio_fp, speed=speech_speed)

    else:
        command = st.text_input("Type your command")
        if st.button("Send"):
            if command:
                st.write(f"You typed: {command}")
                
                if input_lang != 'English':
                    command = translate_text(command, 'en')
                
                response = process_command(command)
                
                if output_lang != 'English':
                    response = translate_text(response, languages[output_lang])
                
                st.write("Assistant:", response)
                audio_fp = text_to_speech(response, languages[output_lang])
                play_audio(audio_fp, speed=speech_speed)

if __name__ == "__main__":
    main()