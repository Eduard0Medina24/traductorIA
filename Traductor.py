import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
config=dotenv_values(".env")

ELEVENLABS_API_KEY= config("ELEVENLABS_API_KEY")



def Translator (audio_file):
    try:
        model=whisper.load_model("base")
        result=model.transcribe(audio_file, language= "Spanish",fp16=False)
        transcription= result ["text"]
    except Exception as e:
        raise gr.Error(
            f"se produjo un error en la transcripcion del texto{str[e]}")
    
    print(f"texto original: {transcription}")
    
    try:
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(
         f"se produjo un error en la traduccion del texto{str[e]}")   
    
    print(f"texto traducido: {en_transcription}")
        
    



    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text= en_transcription,
            model_id="eleven_turbo_v2",  # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),)
    save_file_path= "audios/en.mp3"

    with open (save_file_path,"wb") as f:
        for chunk in response :
            if chunk:
                f.write(chunk)
        
    return chunk    


web=gr.Interface(
    fn=Translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
        ),
    outputs=[gr.Audio(label="ingles")],
    title="traductor de voz",
    description="traductor de idiomas con IA"
).launch

