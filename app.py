from flask import Flask, request, jsonify

import numpy as np
import tensorflow as tf
from transformers import TFAlbertModel

from pre_processing import preprocess_text
from tokenizer import tokenize

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    "promt_model.h5",
    custom_objects={"TFAlbertModel": TFAlbertModel}
)

def display_classes(predicted_class, probability):
    if predicted_class is not None:
        predicted_data = {
            "predicted_class": predicted_class,
            "probability": float(probability),
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
    prom_input_ids, prom_attention_masks = tokenize(processed_promt, 512)

    promt_class = model.predict([prom_input_ids, prom_attention_masks])

    # Nama kelas
    class_names = ['labolatorium_fisika', 'labolatorium_kimia']

    # Prediksi probabilitas
    probabilities = promt_class[0]  

    # Indeks kelas dengan probabilitas tertinggi
    predicted_index = np.argmax(probabilities)

    # Nama kelas
    predicted_class_name = class_names[predicted_index]

    # Probabilitas kelas yang diprediksi
    predicted_probability = probabilities[predicted_index]
    
    return display_classes(predicted_class_name, predicted_probability)

if __name__ == '__main__':
    app.run(debug=True)