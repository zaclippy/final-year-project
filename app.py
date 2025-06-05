from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load your trained models
try:
    # Load custom NER model
    nlp = spacy.load("./output/model-best")
    print("✅ Custom NER model loaded")
except:
    # Fallback to Spanish base model
    nlp = spacy.load("es_core_news_sm")
    print("⚠️ Using Spanish base model")

# Load classification models
try:
    classifier = joblib.load('./classifier.joblib')
    vectorizer = joblib.load('./vectorizer.joblib')
    print("✅ Classification models loaded")
except:
    print("❌ Classification models not found")
    classifier = None
    vectorizer = None

@app.route('/api/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        clinical_text = data.get('text', '')
        
        if not clinical_text:
            return jsonify({'error': 'No text provided'}), 400
        
        if classifier is None or vectorizer is None:
            # Return demo prediction for showcase
            demo_prediction = {
                'prediction': 'Cancer Case' if 'osteosarcoma' in clinical_text.lower() or 'carcinoma' in clinical_text.lower() else 'Non-Cancer',
                'confidence': 0.85,
                'probabilities': {'cancer': 0.85, 'non_cancer': 0.15}
            }
            return jsonify(demo_prediction)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([clinical_text])
        
        # Make prediction
        prediction = classifier.predict(text_vectorized)[0]
        confidence = classifier.predict_proba(text_vectorized)[0].max()
        probabilities = classifier.predict_proba(text_vectorized)[0]
        
        result = {
            'prediction': 'Cancer Case' if prediction == 1 else 'Non-Cancer',
            'confidence': float(confidence),
            'probabilities': {
                'cancer': float(probabilities[1]),
                'non_cancer': float(probabilities[0])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ner', methods=['POST'])
def extract_entities():
    try:
        data = request.get_json()
        clinical_text = data.get('text', '')
        
        if not clinical_text:
            return jsonify({'error': 'No text provided'}), 400
        
        doc = nlp(clinical_text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': get_entity_description(ent.label_)
            })
        
        # Generate HTML for visualization
        html_visualization = generate_entity_html(clinical_text, entities)
        
        result = {
            'entities': entities,
            'html': html_visualization,
            'total_entities': len(entities)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo', methods=['GET'])
def get_demo_data():
    """Return demo cases for showcasing"""

    demo_cases = []
    demo_folder = './demo_txt'
    if os.path.exists(demo_folder):
        for filename in os.listdir(demo_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(demo_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Simple label inference from filename
                if 'cancer' in filename.lower():
                    label = 'Cancer Case'
                elif 'bg' in filename.lower() or 'background' in filename.lower():
                    label = 'Non-Cancer'
                else:
                    label = 'Unknown'
                demo_cases.append({
                    'filename': filename,
                    'text': text,
                    'label': label
                })
    else:
        demo_cases = [
            {'filename': 'example_cancer.txt', 'text': 'Paciente con diagnóstico de osteosarcoma...', 'label': 'Cancer Case'},
            {'filename': 'example_bg.txt', 'text': 'Paciente sin antecedentes oncológicos...', 'label': 'Non-Cancer'}
        ]
    
    return jsonify({'demo_cases': demo_cases})

def get_entity_description(label):
    """Get human-readable description for entity labels"""
    descriptions = {
        'CONDITION': 'Medical condition or disease',
        'SYMPTOM': 'Clinical symptom',
        'TEST': 'Medical test or procedure',
        'FINDING': 'Clinical finding',
        'ANATOMICAL': 'Anatomical location',
        'BACKGROUND': 'Patient background information'
    }
    return descriptions.get(label, 'Medical entity')

def generate_entity_html(text, entities):
    """Generate HTML with highlighted entities"""
    if not entities:
        return f'<p>{text}</p>'
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    
    html_parts = []
    last_end = 0
    
    for entity in sorted_entities:
        # Add text before entity
        html_parts.append(text[last_end:entity['start']])
        
        # Add highlighted entity
        color_map = {
            'CONDITION': '#ff6b6b',
            'SYMPTOM': '#4ecdc4',
            'TEST': '#45b7d1',
            'FINDING': '#96ceb4',
            'ANATOMICAL': '#feca57',
            'BACKGROUND': '#a29bfe'
        }
        color = color_map.get(entity['label'], '#gray')
        
        html_parts.append(
            f'<span class="entity" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px;" '
            f'title="{entity["description"]}">{entity["text"]} <small>[{entity["label"]}]</small></span>'
        )
        
        last_end = entity['end']
    
    # Add remaining text
    html_parts.append(text[last_end:])
    
    return ''.join(html_parts)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'NLP Cancer Diagnosis API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)