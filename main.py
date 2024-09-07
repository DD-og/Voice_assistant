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
import groq
import pygame
import json
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT

GROQ_API_KEY = "your-groq-qpi-key" 

client = groq.Groq(api_key=GROQ_API_KEY)

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

@st.cache_data
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def play_audio(audio_fp, speed=1.3):
    pygame.mixer.init()
    audio = AudioSegment.from_file(audio_fp, format="mp3")
    faster_audio = audio.speedup(playback_speed=speed)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        faster_audio.export(temp_audio.name, format="mp3")
        pygame.mixer.music.load(temp_audio.name)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        os.unlink(temp_audio.name)

def record_audio(duration=5, sample_rate=16000):
    st.write("Recording... Speak now.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
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
    try:
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
            model="llama-3.1-70b-versatile",
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {str(e)}. Please try again later.")
        return "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Please try again later."

def save_conversation(filename):
    with open(filename, 'w') as f:
        json.dump(st.session_state.conversation_history, f)

def load_conversation(filename):
    with open(filename, 'r') as f:
        st.session_state.conversation_history = json.load(f)

def export_conversation(format='txt'):
    content = ""
    for i, (question, answer) in enumerate(st.session_state.conversation_history):
        content += f"Q{i+1}: {question}\n"
        content += f"A{i+1}: {answer}\n\n"
    
    if format == 'txt':
        return content
    elif format == 'pdf':
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        styles.add(ParagraphStyle(name='Question', parent=styles['Normal'], alignment=TA_RIGHT))
        styles.add(ParagraphStyle(name='Answer', parent=styles['Normal']))

        for i, (question, answer) in enumerate(st.session_state.conversation_history):
            story.append(Paragraph(f"Q{i+1}: {question}", styles['Question']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"A{i+1}: {answer}", styles['Answer']))
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        return buffer

def process_and_respond(command, input_lang, output_lang, speech_speed, languages):
    if input_lang != 'English':
        command = translate_text(command, 'en')
    
    response = process_command(command)
    
    if output_lang != 'English':
        response = translate_text(response, languages[output_lang])
    
    st.session_state.conversation_history.append((command, response))
    
    audio_fp = text_to_speech(response, languages[output_lang])
    
    return response, audio_fp

def display_conversation_history(languages, output_lang, speech_speed):
    st.subheader("Conversation History")
    for i, (question, answer) in enumerate(st.session_state.conversation_history):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://img.icons8.com/color/48/000000/user.png", width=50)
        with col2:
            st.markdown(f"<div style='background-color: #E6F3FF; padding: 10px; border-radius: 10px; text-align: right;'>{question}</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<div style='background-color: #F0F0F0; padding: 10px; border-radius: 10px;'>{answer}</div>", unsafe_allow_html=True)
        with col2:
            st.image("https://img.icons8.com/color/48/000000/bot.png", width=50)
        
        if st.button("Play Response", key=f"play_{i}"):
            audio_fp = text_to_speech(answer, languages[output_lang])
            play_audio(audio_fp, speed=speech_speed)
        
        st.write("---")

def main():
    st.title("Dev's Voice Assistant 🚀")

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
                response, audio_fp = process_and_respond(command, input_lang, output_lang, speech_speed, languages)
                st.write("Assistant:", response)
                play_audio(audio_fp, speed=speech_speed)
    else:
        command = st.text_input("Type your command")
        if st.button("Send"):
            if command:
                st.write(f"You typed: {command}")
                response, audio_fp = process_and_respond(command, input_lang, output_lang, speech_speed, languages)
                st.write("Assistant:", response)
                play_audio(audio_fp, speed=speech_speed)

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")

    with col2:
        if st.button("Save Conversation"):
            save_conversation("conversation.json")
            st.success("Conversation saved!")

    with col3:
        if st.button("Load Conversation"):
            load_conversation("conversation.json")
            st.success("Conversation loaded!")

    with col4:
        if st.button("Export as TXT"):
            exported_text = export_conversation(format='txt')
            b64 = base64.b64encode(exported_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="conversation_export.txt">Download TXT</a>'
            st.markdown(href, unsafe_allow_html=True)

    with col5:
        if st.button("Export as PDF"):
            pdf_buffer = export_conversation(format='pdf')
            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="conversation_export.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

    display_conversation_history(languages, output_lang, speech_speed)

if __name__ == "__main__":
    main()
