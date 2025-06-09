import io
import os
import torch
import torchaudio
from dotenv import load_dotenv
from chatterbox.tts import ChatterboxTTS
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# Load environment variables
load_dotenv()
prompt_file = os.getenv('AUDIO_PROMPT', 'sample.mp3')
audio_prompt_path = os.path.join('audio_prompt', prompt_file)
exaggeration = float(os.getenv('EXAGGERATION', 0.5))
cfg_weight = float(os.getenv('CFG_WEIGHT', 0.5))
temperature = float(os.getenv('TEMPERATURE', 0.5))

# Initialize the Chatterbox TTS model
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = ChatterboxTTS.from_pretrained(device=device)
if os.path.isfile(audio_prompt_path):
    model.prepare_conditionals(wav_fpath=audio_prompt_path, exaggeration=exaggeration)
else:
    if prompt_file:
        print(f"Audio prompt file '{audio_prompt_path}' not found. Use default voice.")

# Endpoint for text-to-speech
@app.route('/v1/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    try:
        # Generate audio from text
        buffer = io.BytesIO()
        wav = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
        torchaudio.save(buffer, wav, model.sr, format='wav')
        buffer.seek(0)
        
        # Return the audio file as a response
        return Response(buffer.read(), mimetype='audio/wav', headers={
            'Content-Disposition': 'attachment; filename=output.wav'
        })
    except Exception as e:
        print("TTS generation failed.")
        return jsonify({'error': f'TTS generation failed: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
