import os
import sys
from groq import Groq
import panel as pn
from time import sleep
import speech_recognition as sr
import numpy as np
import openai as op
import requests

import sounddevice as sd
import scipy.io.wavfile as wav
import queue
from os import path

from tts_G import text_to_speech

import json
from link_scraper import get_relavent_text

from NLU_module.main import NLU

from typing import Generator
from chromadb import PersistentClient
# panel serve healthApp.py --static-dirs assets=./assets 


pn.extension()
stream = None
is_recording = False
with_history = True
isRag = True
isNarrator = True
isLocal = True
audio_queue = queue.Queue()

from openai import AzureOpenAI





with open('nhs_illnesses.json', 'r') as f:
    illness_dict = json.load(f)

def search_nhs_knowledgebase(query):
    query_lower = query.lower()
    for letter, illness_list in illness_dict.items():
        for name, url in illness_list:
            if name.lower() in query_lower:
                print(f"[RAG] Matched illness: {name}")
                data = get_relavent_text(url)
                sections = data['content']
                context = ""
                for sec in sections:
                    context += f"{sec['heading']}:\n" + " ".join(sec['text']) + "\n\n"
                return context.strip()
    return 

nlu_agent = NLU(log_folder="log", file_name="rag_llm_log", with_verifier=True)


def get_response(contents, user, instance):
    RAG = RAGModule('./data')
    if isRag:
        print("With RAG.")
        context = RAG.prepare_prompt(contents)
        context = context.encode('gbk', errors='ignore').decode('gbk')
    else:
        print("Without RAG.")
        context = ' '
    
    print(context)
    
    if with_history:
        print('with_history: True')
    result = nlu_agent.run(contents, context, with_history)

    for i in range(len(result)):
        yield result[:i+1]
        sleep(0.03)
    
    if isNarrator:
        if user == "User" and switch3.value:
            text_to_speech(result, lang='en', play_audio=True)

    return result


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

    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), pcm_filename)
    print(AUDIO_FILE)
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source) 

    print("Hi")
    
    if (isLocal):
        speech=r.recognize_whisper(audio, language="english")
    else:
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ")
        
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

def history_mode(value):
    global pn, layout, chat_bot, with_history
    if value.new:
        with_history = True
        print("History mode enabled.")
    else:
        with_history = False
        print("History mode disabled.")

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

def local_mode(value):
    global isLocal
    if value.new:
        isLocal = True
    else:
        isLocal = False
 
button2 = pn.widgets.Button(name="≡")
button = pn.widgets.Button(name="🎤")
button3 = pn.widgets.Button(name="Stop Recording")
button3.button_type = "danger"
switch = pn.widgets.Switch(name="RAG Record", value=True)
switch.param.watch(rag_mode, 'value', onlychanged=True)
switch2 = pn.widgets.Switch(name="History", value=True)
switch2.param.watch(history_mode, 'value', onlychanged=True)
switch3 = pn.widgets.Switch(name="Narrator", value=True)
button.button_type = "primary"
switch4 = pn.widgets.Switch(name="Local", value=True)
switch4.param.watch(local_mode, 'value', onlychanged=True)
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
    pn.widgets.StaticText(name="With History"),
    switch2,
    pn.widgets.StaticText(name="Local"),
    switch4,
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


chat_bot.send("What do you want to know about?", user="Assistant", respond=False)


class RAGModule():
    def __init__(self, path: str):
        COLLECTION_NAME = 'test_collection'
        self.collection = self.initialise_db(path, COLLECTION_NAME)
        self.next_id = self.next_id_generator()

    def initialise_db(self, path: str, collection_name: str):
        """
        Either get or create collection with name `collection_name` in directory `path`.
        """
        chroma_client = PersistentClient(path)
        if collection_name in chroma_client.list_collections():
            # print(f"Collection `{collection_name}` found!")
            return chroma_client.get_collection(collection_name)
        # print(f"Created collection `{collection_name}`")
        return chroma_client.create_collection(collection_name)

    def next_id_generator(self) -> Generator[None, None, int]:
        count = 0
        while True:
            yield count
            count += 1

    def add_to_datastore(self, fact: str, illness_name: str):
        self.collection.add(documents=fact, ids=f'fact_{next(self.next_id)}', metadatas={'illness_name' : illness_name})

    def get_context(self, query: str, num_retrievals = 3) -> str:
        """
        Get relevant documents formatted into paragraph.
        """
        documents = self.collection.query(query_texts=query, n_results=num_retrievals).get("documents", [[]]) # Extract documents from our query
        return "\n".join(documents[0]) # Only return first batch results (single query)

    def prepare_prompt(self, inp):
        #while True:
        return self.get_context(inp)


    