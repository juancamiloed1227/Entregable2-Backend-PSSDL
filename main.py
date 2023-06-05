# Fast API Imports
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
load_dotenv()

# Transcription Imports
from gradio_client import Client
import yt_dlp as youtube_dl
import os
import subprocess
from ffmpeg import Error as FFmpegError
from ffmpeg import run as ffmpeg_run
from io import BytesIO
import openai
import math
from functools import wraps

# Summary imports
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate

API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/" # Whisper Jax HugginFace Space Link
client = Client(API_URL)

openai_api_key = os.getenv('OPENAI_API_KEY') # OpenAI API Key

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Video(BaseModel):
    url: str

# Split audio for Whisper Jax
def split_audio(file, segment_length_ms=20*60*1000):
    try:
        audio = AudioSegment.from_file(file)
        segments = []

        for i in range(0, len(audio), segment_length_ms):
            segments.append(audio[i:i + segment_length_ms])

        return segments
    except CouldntDecodeError:
        print("No se pudo decodificar el archivo. Asegúrate de que sea un archivo de audio válido.")
        return []

# Split audio for Whisper OpenAI
def split_audio_openai(file_path):
    CHUNK_SIZE = 24 * 1024 * 1024
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)  # duration in milliseconds
    chunk_duration = math.ceil((CHUNK_SIZE * 1000) / (audio.frame_rate * audio.frame_width))  # in milliseconds
    return [audio[i:i+chunk_duration] for i in range(0, duration, chunk_duration)]

# Transcribe audio with Whisper OpenAI
def transcribe_audio_chunks(audio_chunks):
    print('Transcribing with Whisper OpenAI')
    transcripts = []
    for chunk in audio_chunks:
        with open("temp_chunk.wav", "wb") as f:
            chunk.export(f)
        with open("temp_chunk.wav", "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        transcripts.append(transcript)
    os.remove("temp_chunk.wav")
    return transcripts

# Transcribe audio with Whisper Jax
def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    print('Transcribing with Whisper Jax')
    """Function to transcribe an audio file using the Whisper JAX endpoint."""
    if task not in ["transcribe", "translate"]:
        raise ValueError("task should be one of 'transcribe' or 'translate'.")

    text, runtime = client.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    os.remove(audio_path)

    return text

# Traanscribe Segments
def transcribe_segments(segments, return_timestamps=False):
    transcription = ""
    for i, segment in enumerate(segments):
        segment_path = f"temp_segment_{i}.mp3"
        segment.export(segment_path, format="mp3")

        try:
            transcribed_text = transcribe_audio(segment_path, return_timestamps=return_timestamps)
        except Exception as e:
            print(f"Error en la transcripción de Whisper JAX: {e}. Cambiando a OpenAI.")
            audio_chunks = split_audio_openai(segment_path)
            transcript_objects = transcribe_audio_chunks(audio_chunks)
            transcribed_text = "\n".join([transcript.text for transcript in transcript_objects])
            
        transcription += transcribed_text + "\n"

    return transcription

def generate_summary(transcription):
    text_splitter = CharacterTextSplitter(separator = ".", chunk_size = 1000, chunk_overlap  = 100, length_function = len)

    text = transcription
    texts = text_splitter.create_documents([text])
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo')

    map_prompt = """
    Escriba un resumen conciso de lo siguiente:
    "{text}"
    RESUMEN CONCISO:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    Escriba un resumen conciso del siguiente texto delimitado por comillas triples.
    Devuelva su respuesta en viñetas que cubran los puntos clave del texto.
    ```{text}```
    RESUMEN EN VIÑETAS:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template, combine_prompt=combine_prompt_template, verbose=True, return_intermediate_steps=True)
    output = summary_chain({"input_documents": texts}, return_only_outputs=True)

    complete_summary = ''
    for sum in output['intermediate_steps']:
        complete_summary += sum + '\n'

    return output['output_text'], complete_summary

@app.post("/api/summary")
async def transcribe_yt_video(video: Video):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "yt_video.%(ext)s",
    }
    
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(video.url, download=False)
            if video_info['duration'] > 1200:
                return {'error': 'El video excede la duración maxima permitida (20 Min)'}
            
            video_title = video_info['title']
            channel = video_info['uploader']
            thumbnail = video_info['thumbnail']
            video_id = video_info['id']
            iframe_code = f'https://www.youtube.com/embed/{video_id}'
            ydl.download([video.url])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al descargar el video: {e}")
    
    try:
        audio_segments = split_audio("yt_video.mp3")
        os.remove("yt_video.mp3")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al dividir el archivo de audio: {e}")
    
    try:
        output_with_timestamps = transcribe_segments(audio_segments, return_timestamps=False)
        response = f'Titulo: {video_title}\nAutor: {channel}\n\n{output_with_timestamps}'
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al transcribir el audio: {e}")

    try:
        kp_sum, complete_sum = generate_summary(response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al generarl el resumen: {e}")

    return {
        "title": video_title,
        "channel": channel,
        "thumbnail": thumbnail,
        "iframe_code": iframe_code,
        "keypoints":kp_sum,
        "complete": complete_sum,
        "transcription": response
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)