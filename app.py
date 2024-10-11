from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS if frontend and backend are on different origins

# Load your pickle file (e.g., a model for video identification)
with open('video_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/identify-video', methods=['POST'])
def identify_video():
    data = request.json
    # Assume the data contains the necessary input for the model
    # For example, features extracted from the video
    result = model.predict([data['features']])
    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
