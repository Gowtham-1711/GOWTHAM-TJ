# GOWTHAM-TJ
SEAMLESS M4T-POWERED SPEECH TO TEXT WITH GRADIO UI
Description:
This project aims to create a speech-to-text translation system using the SEAMLESS M4T model, a foundational multilingual and multitask model developed by Facebook Research. The system allows users to translate speech input into text in multiple languages. The translation is powered by the SEAMLESS M4T model and is presented through a user-friendly interface built using the Gradio library.

Features
Supports 36 languages for speech input, including Arabic, Chinese, French, German, Hindi, Japanese, Korean, Russian, Spanish, and more.
Supports English and Tamil for text output.
Uses a state-of-the-art multitask model that can perform speech recognition, speech translation, and text translation.
Uses a high-quality vocoder to synthesize speech from text.
Uses Gradio to create a user-friendly web interface that allows users to upload audio files or record audio from the microphone.

Installation
To run this project, you need to have Python 3.6 or higher and a CUDA-enabled GPU. You also need to install the following dependencies:
!pip install torch==2.0.1
!pip install fairseq2==0.1 pydub
!pip install gradio==3.50.0
!pip install soundfile
!pip install librosa

You also need to clone the seamless communication repository and install it:
!git clone https://github.com/facebookresearch/seamless_communication.git
%cd seamless_communication
!git checkout 01c1042841f9bce66902eb2c7512dbdd71d42112
!pip install .

Usage
To run the project, you need to import the necessary modules and initialize a Translator object with the seamlessM4T_large model and the vocoder_36langs vocoder:

import gradio as gr
import torch
from seamless_communication.models.inference import Translator
import librosa
import os
import soundfile as sf
# Initialize a Translator object with a multitask model and vocoder on the GPU
translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"))

Then, you need to define two helper functions: one to downsample the input audio file to 16000 Hz, and another to perform the speech-to-text translation using the Translator object

Python
def downsample_audio(input_audio_file, output_audio_file, target_sr):
    audio_file, sr = librosa.load(input_audio_file, sr=None)
    audio_file_downsampled = librosa.resample(audio_file, orig_sr=sr, target_sr=16000)
    sf.write(output_audio_file, audio_file_downsampled, 16000)
def speech_to_text(input_audio_mic, input_audio_file, tgt_lang):
    # Check the input audio source and convert it to 16-bit integer format if needed
    if input_audio_mic is not None:
        input_audio_mic_path = input_audio_mic
    elif input_audio_file is not None:
        data, sr = sf.read(input_audio_file)
        sf.write("output.wav", data, sr, format='WAV', subtype='PCM_16')
        input_audio_file = "output.wav"
        input_audio_mic_path = input_audio_file
    else:
        return "Please provide an audio input."
    try:
        # Downsample the input audio file to 16000 Hz
        downsampled_audio_file = "downsampled_output.wav"
        downsample_audio(input_audio_mic_path, downsampled_audio_file, 16000)
        with open(downsampled_audio_file, "rb") as file:
            # Predict the translated text using seamless communication model
            translated_text, _, _ = translator.predict(file.name, "s2tt", tgt_lang)
            # Return the translated text as output
            return translated_text
    except Exception as e:
        # Return an error message if any exception occurs during translation
        return f"Error translating {input_audio_mic_path}: {e}"

Finally, you need to create the UI components for the audio input, the target language selection, and the translated text output, and create an interface object with the speech_to_text function and the UI components. You can also customize the title and description of the interface:
Python
# Create UI components for audio input, target language selection and translated text output
audio_input_mic = gr.Audio(source="microphone", type="filepath", label="Input Audio from Microphone")
audio_input_file = gr.Audio(source="upload", type="filepath", label="Input Audio from File")
tgt_lang = gr.Dropdown(choices=[("English", "eng"), ("Tamil", "tam")], label="Target Language")
translated_text_output = gr.Textbox(label="Translated Text")

# Create an interface object with speech_to_text function and UI components
demo = gr.Interface(
    fn=speech_to_text,
    inputs=[audio_input_mic, audio_input_file, tgt_lang],
    outputs=translated_text_output,
    title="SEAMLESS M4T-POWERED SPEECH TO TEXT WITH GRADIO UI",
    description="Translate Audio",
)

# Launch the interface in a web browser
demo.launch()

To launch the interface, run the last line of code in your Python environment. You should see a web page like this:

You can then choose to upload an audio file or record an audio from the microphone, select the target language, and click the submit button to see the translated text.

References
Seamless Communication: A Multitask Model for Speech Recognition, Translation, and Synthesis
[Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild]
