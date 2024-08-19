import gradio as gr

def Translator (audio_file):
    try:
        model=whisper.load_model("base")
        result=model.transcribe(audio_file, language= "Spanish",fp16=False)
        transcription= result ["text"]
    except Exception as e:
        raise gr.Error(
            f"se produjo un error en la transcripcion del texto{str[e]}")
    
    print(f"texto original: {transcription}")



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