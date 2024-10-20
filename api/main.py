from flask import Flask, request, jsonify
import os
import requests
import tempfile
import google.generativeai as genai

# Configurer l'API Gemini avec votre clé API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionnaire pour stocker les historiques de conversation
sessions = {}

def download_file(url):
    """Télécharge un fichier depuis une URL et retourne le chemin du fichier temporaire."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Déterminer le type de fichier et suffixe approprié
        suffix = os.path.splitext(url)[1]  # Obtenir l'extension du fichier
        valid_suffixes = ['.pdf', '.docx', '.doc', '.html', '.txt']
        if suffix not in valid_suffixes:
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            return temp_file.name
    else:
        return None

def upload_to_gemini(path, mime_type=None):
    """Télécharge le fichier donné sur Gemini."""
    if not mime_type:
        mime_type = "application/octet-stream"  # Type MIME par défaut
    
    # Définir correctement le type MIME pour les formats pris en charge
    if path.endswith('.pdf'):
        mime_type = "application/pdf"
    elif path.endswith('.docx'):
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif path.endswith('.doc'):
        mime_type = "application/msword"
    elif path.endswith('.html'):
        mime_type = "text/html"
    elif path.endswith('.txt'):
        mime_type = "text/plain"

    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Configuration du modèle avec les paramètres de génération
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/api/gemini', methods=['POST'])
def handle_request():
    try:
        data = request.json
        prompt = data.get('prompt', '')  # Question ou prompt de l'utilisateur
        custom_id = data.get('customId', '')  # Identifiant de l'utilisateur ou session
        file_url = data.get('link', '')  # URL du fichier

        # Récupérer l'historique de la session existante ou en créer une nouvelle
        if custom_id not in sessions:
            sessions[custom_id] = []  # Nouvelle session
        history = sessions[custom_id]

        # Télécharger et ajouter le fichier à l'historique s'il est présent
        if file_url:
            file_path = download_file(file_url)
            if file_path:
                mime_type = "application/octet-stream"  # Type MIME par défaut
                if file_path.endswith('.pdf'):
                    mime_type = "application/pdf"
                elif file_path.endswith('.docx'):
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif file_path.endswith('.doc'):
                    mime_type = "application/msword"
                elif file_path.endswith('.html'):
                    mime_type = "text/html"
                elif file_path.endswith('.txt'):
                    mime_type = "text/plain"
                
                file = upload_to_gemini(file_path, mime_type)
                if file:
                    history.append({
                        "role": "user",
                        "parts": [file, prompt],
                    })
                else:
                    return jsonify({'message': 'Failed to upload file to Gemini'}), 500
            else:
                return jsonify({'message': 'Failed to download file'}), 500
        else:
            history.append({
                "role": "user",
                "parts": [prompt],
            })

        # Démarrer ou continuer une session de chat avec l'historique
        chat_session = model.start_chat(history=history)

        # Envoyer un message dans la session de chat
        response = chat_session.send_message(prompt)

        # Ajouter la réponse du modèle à l'historique
        history.append({
            "role": "model",
            "parts": [response.text],
        })

        # Retourner la réponse du modèle
        return jsonify({'message': response.text})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'message': 'Internal Server Error'}), 500

@app.route('/')
def home():
    return '<h1>Votre API Gemini est en cours d\'exécution...</h1>'

if __name__ == '__main__':
    # Héberger l'application Flask sur 0.0.0.0 pour qu'elle soit accessible publiquement
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
