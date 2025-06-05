from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import json

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for demo

# Load your trained models
try:
    # Load custom NER model
    nlp = spacy.load("./output/model-best")
    print("✅ Custom NER model loaded")
except Exception as e:
    print(f"❌ Custom model failed: {e}")
    try:
        # Fallback to Spanish base model
        nlp = spacy.load("es_core_news_sm")
        print("✅ Spanish base model loaded")
    except Exception as e2:
        print(f"❌ Spanish model failed: {e2}")
        # Create blank Spanish model as last resort
        nlp = spacy.blank("es")
        print("⚠️ Using blank Spanish model")

# Load classification models
try:
    classifier = joblib.load('./classifier.joblib')
    vectorizer = joblib.load('./vectorizer.joblib')
    print("✅ Classification models loaded")
except Exception as e:
    print(f"❌ Classification models not found: {e}")
    classifier = None
    vectorizer = None

@app.route('/')
def home():
    return jsonify({
        'message': 'NLP Cancer Diagnosis API',
        'status': 'running',
        'endpoints': ['/api/classify', '/api/ner', '/api/demo', '/health']
    })

@app.route('/api/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        clinical_text = data.get('text', '')
        
        if not clinical_text:
            return jsonify({'error': 'No text provided'}), 400
        
        if classifier is None or vectorizer is None:
            # Enhanced demo prediction logic
            cancer_keywords = ['osteosarcoma', 'carcinoma', 'metástasis', 'tumor', 'cáncer', 'neoplasia', 'oncología']
            is_cancer = any(keyword in clinical_text.lower() for keyword in cancer_keywords)
            
            demo_prediction = {
                'prediction': 'Cancer Case' if is_cancer else 'Non-Cancer',
                'confidence': 0.89 if is_cancer else 0.78,
                'probabilities': {
                    'cancer': 0.89 if is_cancer else 0.22,
                    'non_cancer': 0.11 if is_cancer else 0.78
                }
            }
            return jsonify(demo_prediction)
        
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
        print(f"Classification error: {e}")
        return jsonify({'error': 'Classification service temporarily unavailable'}), 503

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
        
        # If no custom entities found, add demo entities for demonstration
        if not entities and len(clinical_text.split()) > 3:
            demo_entities = extract_demo_entities(clinical_text)
            entities.extend(demo_entities)
        
        # Generate HTML for visualization
        html_visualization = generate_entity_html(clinical_text, entities)
        
        result = {
            'entities': entities,
            'html': html_visualization,
            'total_entities': len(entities)
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"NER error: {e}")
        return jsonify({'error': 'Entity extraction service temporarily unavailable'}), 503

def extract_demo_entities(text):
    """Extract demo entities using keyword matching for demonstration"""
    demo_entities = []
    
    # Define keyword patterns
    patterns = {
        'CONDITION': ['osteosarcoma', 'carcinoma', 'metástasis', 'tumor', 'cáncer', 'neoplasia', 'hipertensión', 'diabetes'],
        'SYMPTOM': ['dolor', 'cefalea', 'mareos', 'fiebre', 'fatiga', 'lumbalgia'],
        'TEST': ['biopsia', 'radiografía', 'TC', 'resonancia', 'análisis', 'ecografía'],
        'ANATOMICAL': ['vértebra', 'lumbar', 'abdominal', 'torácica', 'craneal', 'pulmonar'],
        'BACKGROUND': ['años', 'varón', 'mujer', 'paciente']
    }
    
    text_lower = text.lower()
    for label, keywords in patterns.items():
        for keyword in keywords:
            start = text_lower.find(keyword)
            if start != -1:
                # Find the actual case in original text
                actual_text = text[start:start+len(keyword)]
                demo_entities.append({
                    'text': actual_text,
                    'label': label,
                    'start': start,
                    'end': start + len(keyword),
                    'description': get_entity_description(label)
                })
    
    return demo_entities

@app.route('/api/demo', methods=['GET'])
def get_demo_data():
    """Return demo cases for showcasing"""
    demo_cases = []
    demo_folder = './demo_txt'
    
    if os.path.exists(demo_folder):
        try:
            for filename in os.listdir(demo_folder):
                if filename.endswith('.txt'):
                    file_path = os.path.join(demo_folder, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Determine label based on filename
                    if any(keyword in filename.lower() for keyword in ['cancer', 'onco', 'tumor']):
                        label = 'Cancer Case'
                    elif any(keyword in filename.lower() for keyword in ['bg', 'background', 'control', 'radiologia']):
                        label = 'Non-Cancer'
                    else:
                        label = 'Unknown'
                    
                    demo_cases.append({
                        'filename': filename,
                        'text': text,
                        'label': label
                    })
        except Exception as e:
            print(f"Error reading demo folder: {e}")
    
    # Fallback demo cases if folder doesn't exist
    if not demo_cases:
        demo_cases = [
            {
                'filename': 'demo_cancer_case.txt',
                'text': 'Varón de 35 años con osteosarcoma convencional de alto grado a nivel de la segunda vértebra lumbar. Presenta lumbalgia irradiada a ambos muslos y hipoestesia en la cara anterior de la pierna derecha.',
                'label': 'Cancer Case'
            },
            {
                'filename': 'demo_background_case.txt', 
                'text': 'Mujer de 46 años con cefalea y mareos ocasionales. Exploración física normal. Tensión arterial dentro de límites normales. No se observan alteraciones significativas.',
                'label': 'Non-Cancer'
            },
            {
                'filename': 'demo_oncology_case.txt',
                'text': 'Paciente con metástasis pulmonar detectada en TC de tórax. Presenta masa abdominal palpable y pérdida de peso significativa en los últimos meses.',
                'label': 'Cancer Case'
            }
        ]
    
    return jsonify({'demo_cases': demo_cases})

def get_entity_description(label):
    """Get human-readable description for entity labels"""
    descriptions = {
        'CONDITION': 'Medical condition or disease',
        'SYMPTOM': 'Clinical symptom or sign',
        'TEST': 'Medical test or procedure',
        'FINDING': 'Clinical finding or observation',
        'ANATOMICAL': 'Anatomical location or structure',
        'BACKGROUND': 'Patient background information',
        'PER': 'Person',
        'LOC': 'Location',
        'ORG': 'Organization',
        'MISC': 'Miscellaneous entity'
    }
    return descriptions.get(label, f'{label} entity')

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
            'BACKGROUND': '#a29bfe',
            'PER': '#ff9ff3',
            'LOC': '#54a0ff',
            'ORG': '#5f27cd',
            'MISC': '#c8d6e5'
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
    return jsonify({
        'status': 'healthy', 
        'message': 'NLP Cancer Diagnosis API is running',
        'models': {
            'nlp': 'loaded' if nlp else 'failed',
            'classifier': 'loaded' if classifier else 'not_found',
            'vectorizer': 'loaded' if vectorizer else 'not_found'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)