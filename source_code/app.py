from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the trained model
MODEL_PATH = 'fuzzy_cnn_lstm.keras'  
model = load_model(MODEL_PATH)

# Define Flask app
app = Flask(__name__)

# Class mapping: 0 = Normal, all others = Attack
class_mapping = {
    0: "Normal",
    1: "Attack",
    2: "Attack",
    3: "Attack",
    4: "Attack"
}

# History list to store previous predictions
prediction_history = []

@app.route('/')
def home():
    return render_template('index.html', history=prediction_history)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/flowcharts')
def flowcharts():
    return render_template('flowcharts.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        try:
            # Parse input data
            input_data = request.json
            print(f"Received input data: {input_data}")  # Debug log

            can_id = input_data.get('can_id')
            data_bytes = input_data.get('data')

            # Validate input
            if not can_id or len(data_bytes) != 8:
                return jsonify({'error': 'Invalid input. Provide a valid CAN ID and 8 data bytes.'}), 400

            # Convert CAN ID to integer (hex or decimal)
            can_id_int = int(can_id, 16) if can_id.startswith('0x') else int(can_id)

            # Convert data bytes to integers (hex or decimal)
            parsed_data_bytes = []
            for byte in data_bytes:
                if isinstance(byte, str) and byte.startswith('0x'):
                    parsed_data_bytes.append(int(byte, 16))  # Convert hex to int
                else:
                    parsed_data_bytes.append(int(byte))  # Convert decimal to int

            # Combine CAN ID and data bytes into an input sequence
            input_sequence = [can_id_int] + parsed_data_bytes

            # Predict for each element individually
            predictions = []
            for value in input_sequence:
                input_array = np.array(value, dtype=np.float32).reshape((1, 1, 1))
                prediction = model.predict(input_array)
                predictions.append(prediction)

            # Aggregate predictions (e.g., majority voting)
            aggregated_prediction = np.argmax(np.mean(predictions, axis=0))
            confidence = float(np.max(np.mean(predictions, axis=0)))

            # Map prediction to attack type
            attack_status = "Normal" if aggregated_prediction == 0 else "Attack"

            # Get current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Store prediction in history
            prediction_history.append({
                'timestamp': timestamp,
                'can_id': can_id,
                'data': ' '.join(map(str, parsed_data_bytes)),  # Ensure the data is joined as a string
                'prediction': attack_status,
                'confidence': f"{confidence:.2f}"
            })

            # Return prediction with timestamp
            return jsonify({
                'timestamp': timestamp,
                'prediction': attack_status,
                'confidence': f"{confidence:.2f}",
                'data': {
                    'can_id': can_id,
                    'data_bytes': parsed_data_bytes
                }
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('predictions.html', history=prediction_history)

if __name__ == '__main__':
    app.run(debug=True)