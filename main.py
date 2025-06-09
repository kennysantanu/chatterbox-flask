import io
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# Local variables
audio_prompt_path = ""
exaggeration = 0.7
cfg_weight = 0.9
temperature = 0.5

# Initialize the Chatterbox TTS model
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = ChatterboxTTS.from_pretrained(device=device)
if audio_prompt_path:
    model.prepare_conditionals(wav_fpath=audio_prompt_path, exaggeration=exaggeration)

# Endpoint for text-to-speech
@app.route('/v1/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']

    # Generate audio from text
    wav = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, model.sr, format='wav')
    buffer.seek(0)

    # Return the audio file as a response
    return Response(buffer.read(), mimetype='audio/wav', headers={
        'Content-Disposition': 'attachment; filename=output.wav'
    })
    
if __name__ == '__main__':
    app.run(debug=True)
