import os
import sys
from threading import Thread
from groq import Groq
import panel as pn
from time import sleep
import speech_recognition as sr
import numpy as np
import openai as op
import pygame as pg
import sys
sys.path.append('./TTS')
sys.path.append('./RAG')
sys.path.append('./LLM')
import tts as tts
from RAGComponent import RAGModule
from LLMComponent import LLMModule


import sounddevice as sd
import scipy.io.wavfile as wav
import queue
from os import path
pn.extension()
stream = None
is_recording = False
isLocal = True
isRag = True
isNarrator = False
audio_queue = queue.Queue()
rag_module = RAGModule('./data')
llm = LLMModule('qwen2.5:0.5b', None, 'http://localhost:11434/v1')

from openai import AzureOpenAI


def pretty_print_contents(input: str):
    print('='*25 + " CONTEXT " + '='*25)
    print(input)
    print('='*59)

# Function that can deal with the response from the chat bot
def get_response(contents, user, instance):
    # contents = The contents of the user sent message
    # response = The response to be sent back to the user
    if isRag and isLocal:
        contents = rag_module.prepare_prompt(contents)
        response = llm.make_query(contents, True, debug=True)
    elif isRag and not isLocal:
        contents = rag_module.prepare_prompt(contents)
        response = "Replace this with your RAG response with a non local model."
    elif isLocal:
        response = llm.make_query(contents, True, debug=True)
    else:
        response = "Replace this with your local model response."
    
    if isNarrator:
        Thread(target=lambda: tts.text_to_speech(response)).start()
        
    
    for index in range(len(response)):
        yield response[0:index+1]
        sleep(0.001)
    return response

def audio_callback(indata, frames, time, status, something):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def test(instance):
    global is_recording, stream, button
    chat_bot.widgets = [button3]
    button3.visible = True
    # button.disabled = True
    # chat_bot.button_properties = {"Hola": {"icon": "🔴", "callback": stop_recording}}
    sd.default.samplerate = 44100
    # Get the default input and output device information
    input_info = sd.query_devices(kind='input')
    output_info = sd.query_devices(kind='output')

    # Set the default channels to match the system's default input/output channels
    sd.default.channels = min(input_info['max_input_channels'], output_info['max_output_channels'])

    all = []
    # myrecording2 = sd.rec(int(duration * sd.default.samplerate), blocking=True)
    stream = sd.Stream(callback=audio_callback, samplerate=sd.default.samplerate, channels=sd.default.channels, dtype='int16', clip_off=False)
    stream.start()
    is_recording = True
    # wait until the user 

    # sd.play(myrecording2, blocking=True)
    # sd.wait()


    return "Test"

def stop_recording(instance):
    global is_recording, stream, button
    button3.disabled = True
    button3.name = "Processing..."
    button3.button_type = "warning"
    if is_recording:
        print("Recording stopped.")
        # Stop the recording by stopping the stream
        is_recording = False
        sd.stop()  # Stop the sounddevice stream
        
        # Collect all audio data into a single numpy array
        frames = []
        while not audio_queue.empty():
            frames.append(audio_queue.get())
        audio_data = np.concatenate(frames, axis=0)
        stream.abort()
    else:
        print("No recording in progress.")
        return None

    
        #writing the audio to a file, duration is not known
    myrecording2 = audio_data
    filename = 'output.wav'
    wav.write(filename, sd.default.samplerate, myrecording2) 
    # Convert the audio to PCM format and save it
    pcm_filename = 'output_pcm.wav'
    # Number is the maximum possible value for a 32 bit integer, replace it with max of 16 bit to use np.int16, normalizing the audio 2147483647
    wav.write(pcm_filename, sd.default.samplerate, (myrecording2 / np.max(np.abs(myrecording2)) * 2147483647).astype(np.int32))
    # wav.write(pcm_filename, sd.default.samplerate, (myrecording2 / np.max(np.abs(myrecording2)) * 32767).astype(np.int16))

    AUDIO_FILE = pcm_filename
    print(AUDIO_FILE)
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source) 

    print("Hi")
    
    if (isLocal):
        speech=r.recognize_whisper(audio, language="english")
    else:
        os.environ["GROQ_API_KEY"] = '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'
        
        speech=r.recognize_groq(audio, model="whisper-large-v3-turbo")
    
    print(speech)
    chat_bot.send(speech, user="User", respond=True)
    
    print("Hi2")
    # Rese the microphone button icon when done
    button3.disabled = False
    button3.name = "Stop Recording"
    button3.button_type = "danger"
    button.visible = True
    button3.visible = False
    chat_bot.widgets = [chatArea, button]
    return "speech"

def toggle_sidebar(instance):
    global sidebar
    if sidebar.visible:
        sidebar.visible = False
    else:
        sidebar.visible = True
   

def dark_mode(value):
    global pn, layout, chat_bot
    if value.new:
        pn.config.theme = 'dark'
    else:
        pn.config.theme = 'default'
        print(pn.config.theme)

def local_mode(value):
    global pn, layout, chat_bot, isLocal
    if value.new:
        isLocal = True
        print("Local mode enabled.")
    else:
        isLocal = False
        print("Local mode disabled.")

def rag_mode(value):
    global pn, layout, chat_bot, isRag
    if value.new:
        isRag = True
        print("RAG mode enabled.")
    else:
        isRag = False
        print("RAG mode disabled.")

def narrator_mode(value):
    global isNarrator
    if value.new:
        isNarrator = True
    else:
        isNarrator = False
    
button2 = pn.widgets.Button(name="≡")
button = pn.widgets.Button(name="🎤")
button3 = pn.widgets.Button(name="Stop Recording")
button3.button_type = "danger"
switch = pn.widgets.Switch(name="RAG Record", value=True)
switch.param.watch(rag_mode, 'value', onlychanged=True)
switch2 = pn.widgets.Switch(name="Local", value=True)
switch2.param.watch(local_mode, 'value', onlychanged=True)
switch3 = pn.widgets.Switch(name="Narrator", value=False)
switch3.param.watch(narrator_mode, 'value', onlychanged=True)
button.button_type = "primary"
button.on_click(test)
button2.on_click(toggle_sidebar)
button3.on_click(stop_recording)
pn.config.theme = 'default'

sidebar = pn.Column(
    pn.pane.Markdown("## Sidebar"),
    pn.widgets.StaticText(name="Narrator"),
    switch3,
    pn.widgets.StaticText(name="RAG Mode"),
    switch,
    pn.widgets.StaticText(name="Local Mode"),
    switch2,
    sizing_mode="stretch_both"
)



pn.extension(design='material', global_css=[':root { --design-primary-color: blue;  --design-secondary-color: white; --design-secondary-text-color: blue; --design-background-color: white; --design-surface-color: blue; --design-surface-text-color: white;}'])
button2.stylesheets = [':host(.solid) .bk-btn.bk-btn-default {font-weight: bold; font-size: 20px;}']
chatArea = pn.chat.ChatAreaInput(placeholder="Enter your question", resizable=False)
chat_bot = pn.chat.ChatInterface(callback=get_response, widgets=[chatArea, button, button3], max_height=500, default_avatars={"System": "S", "User": "👤"}, reaction_icons={"like": "thumb-up"},  message_params={
        "stylesheets": ['assets/style.css'],
    })
chat_bot.widgets = [chatArea, button]
chat_bot.stylesheets = ["bk-input {color: blue;}"]
button3.visible = False

# layout = pn.Row(sidebar, chat_bot)

# chat_bot.append(button)
# chat_bot.append(button2)

main = ""
panelMain = pn.template.FastListTemplate(
    title = "Health App",
    header_background="#E8B0E6",
    main=[chat_bot],
    sidebar=[sidebar],

    sidebar_width=500,
).servable()


chat_bot.send("Ask me what a wind turbine is", user="Assistant", respond=False)
