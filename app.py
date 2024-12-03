from flask import Flask, request, jsonify
from pyngrok import ngrok

import tensorflow as tf
from tensorflow.keras.models import load_model

from pre_processing import preprocess_text
from tokenizer import tokenize
from pre_trained import create_model


app = Flask(__name__)

# Load model
model = load_model("promt_model.h5")

def display_classes(predicted_class, probabilities):
    if predicted_class is not None:
        predicted_data = {
            "predicted_class": predicted_class,
            "probability": max(probabilities),
        }
        return jsonify(
            message="Prompt Prediction Successful",
            category="success",
            data=predicted_data,
            status=200
        )
    else:
        return jsonify(
            message="Prompt Not Found",
            category="danger",
            data=None,
            status=404
        )

@app.route('/predict', methods=['POST'])
def predict_classes():
    promt = request.form.get('promt', '')
    if not promt:
        return jsonify(
            message="No prompt provided",
            category="error",
            data=None,
            status=400
        )

    # Preprocessing and tokenize
    processed_promt = preprocess_text(promt)
    prom_input_ids, prom_attention_masks = tokenize(processed_promt, 768)

    # Predict
    result = model.predict([prom_input_ids, prom_attention_masks])
    probabilities = tf.nn.softmax(result, axis=1).numpy()[0]
    predicted_class = result.argmax(axis=1)[0]
    
    return display_classes(predicted_class, probabilities)

if __name__ == '__main__':
    app.run(debug=True)