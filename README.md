# Dev's Voice Assistant 🚀

A sophisticated multilingual voice assistant powered by Llama 3.1 70B, built with Streamlit. This application offers a seamless multilingual experience, allowing users to interact through voice or text in multiple languages. Dev's Voice Assistant combines cutting-edge AI technology with a user-friendly interface to provide intelligent responses and natural language processing capabilities.


##Flowchart

![mermaid-diagram-2024-09-07-103842](https://github.com/user-attachments/assets/0e62735c-0808-4a33-b765-23d83cf198ab)



## Features

- **Dual Input Methods**: Choose between voice and text input for maximum flexibility.
- **Multilingual Support**: Communicate in various languages for both input and output.
- **Speech-to-Text Conversion**: Accurately transcribes spoken words into text.
- **Text-to-Speech Synthesis**: Converts text responses into natural-sounding speech.
- **Language Translation**: Seamlessly translates between different languages.
- **AI-Powered Responses**: Utilizes Llama 3.1 70B model via Groq API for intelligent and context-aware replies.
- **Adjustable Speech Speed**: Customize the playback speed of audio responses.
- **Conversation Management**: Save, load, and clear conversation history.
- **Export Functionality**: Export conversations in TXT and PDF formats.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive web experience.

## Supported Languages

- English
- French
- Spanish
- German
- Italian
- Japanese
- Hindi
- Gujarati


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/devs-voice-assistant.git
   cd devs-voice-assistant
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv voiceassistant_env
   source voiceassistant_env/bin/activate  # On Windows, use `voiceassistant_env\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Groq API key:
   - Create a `.env` file in the project root
   - Add your API key: `GROQ_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run final_voiceassistant.py
   ```

2. Open the provided URL in your web browser.

3. Select your preferred input and output languages.

4. Choose between voice or text input.

5. If using voice input, click "Start Listening" and speak your command or question.

6. For text input, type your command and click "Send".

7. The assistant will process your input, generate a response, and provide it in your chosen output format (voice and text).

8. Adjust the speech speed slider to customize the playback speed of voice responses.

## Customization

- To add more languages, update the `languages` dictionary in the `main()` function.
- Modify the system prompt in the `process_command()` function to change the assistant's behavior.

## Contributing

Contributions to improve Dev's Voice Assistant are welcome. Please feel free to submit a Pull Request.
