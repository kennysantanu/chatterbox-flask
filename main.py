from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/v1/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    
if __name__ == '__main__':
    app.run(debug=True)
