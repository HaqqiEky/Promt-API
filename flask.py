from flask import Flask, request, jsonify
from pyngrok import ngrok
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
ngrok.set_auth_token("2fv2gezjnqjXnkqvczgW0MgYQNb_3RgWKK76xAFyBzBZopm8p")
public = ngrok.connect(5000).public_url

# Define Model


# Load model


# Image transformation


def display_drug_information(predicted_class, probabilities, image_url):
    disease_info = csv_path[csv_path['Kelas'] == predicted_class]
    if not disease_info.empty:
        disease = disease_info.to_dict('records')[0]
        symptoms = [symptom.strip() for symptom in str(disease['Gejala']).split('-') if str(disease['Gejala']) != 'nan']
        preventions = [prevention.strip() for prevention in str(disease['Penanggulangan']).split('-') if str(disease['Penanggulangan']) != 'nan']
        disease_info_dict = {
            "disease_name": disease['Kelas'],
            "description": disease['Penjelasan'],
            "symptoms": symptoms,
            "causes": disease['Penyebab'],
            "preventions": preventions,
            "probability": max(probabilities),
            "image_url": image_url
        }
        return jsonify(
              message=("Image prediction successful"),
              category="success",
              data=disease_info_dict,
              status=200
            )
    else:
        return jsonify(
                message=_("No disease information found"),
                category="danger",
                data=None,
                status=404
            )

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify(
                message=("No file part"),
                category="error",
                data=None,
                status=404
            )

    file = request.files['file']
    if file.filename == '':
        return jsonify(
                message=("No selected file"),
                category="error",
                data=None,
                status=404
            )

    if file and file.filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']:
      filename = secure_filename(file.filename)
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      image_url = f"{request.host_url}{app.config['UPLOAD_FOLDER']}/{filename}"

      # Ensure the directory exists
      os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

      file.save(filepath)

      img = Image.open(filepath).convert('RGB')
      img_t = test_transform(img)
      img_t = img_t.unsqueeze(0).to(device)  # Add to device

      with torch.no_grad():
          output = model(img_t)
      probabilities = torch.softmax(output, dim=1).tolist()[0]
      _, predicted_idx = torch.max(output, 1)
      predicted_class = ['Coccidiosis', 'Healthy', 'NewCastleDisease', 'Salmonella'][predicted_idx.item()]
      return display_drug_information(predicted_class, probabilities, image_url)

if __name__ == '__main__':
    print(f"access: {public}")
    app.run(port = 5000)