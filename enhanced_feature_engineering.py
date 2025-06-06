import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import re

class MedicalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract domain-specific medical features"""
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model or spacy.load("es_core_news_sm")
        
        # Medical vocabularies for Spanish
        self.cancer_terms = {
            'carcinoma', 'sarcoma', 'adenocarcinoma', 'melanoma', 'linfoma', 
            'metástasis', 'neoplasia', 'tumor', 'cáncer', 'oncología', 'quimioterapia',
            'radioterapia', 'biopsia', 'malignidad', 'invasivo'
        }
        
        self.anatomy_terms = {
            'pulmón', 'hígado', 'páncreas', 'estómago', 'colon', 'mama', 'próstata',
            'riñón', 'vejiga', 'cerebro', 'hueso', 'sangre', 'linfonodo'
        }
        
        self.symptom_terms = {
            'dolor', 'fatiga', 'pérdida', 'masa', 'nódulo', 'inflamación',
            'sangrado', 'tos', 'disnea', 'anemia', 'fiebre'
        }
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            doc = self.nlp(text.lower())
            
            # Feature extraction
            text_features = {
                # Cancer terminology density
                'cancer_term_count': sum(1 for token in doc if token.lemma_ in self.cancer_terms),
                'cancer_term_density': sum(1 for token in doc if token.lemma_ in self.cancer_terms) / len(doc),
                
                # Anatomical references
                'anatomy_term_count': sum(1 for token in doc if token.lemma_ in self.anatomy_terms),
                
                # Symptom mentions
                'symptom_term_count': sum(1 for token in doc if token.lemma_ in self.symptom_terms),
                
                # Text statistics
                'text_length': len(text),
                'sentence_count': len(list(doc.sents)),
                'avg_sentence_length': len(doc) / max(len(list(doc.sents)), 1),
                
                # Medical entity counts (using your trained NER)
                'condition_entities': len([ent for ent in doc.ents if ent.label_ == 'CONDITION']),
                'symptom_entities': len([ent for ent in doc.ents if ent.label_ == 'SYMPTOM']),
                'test_entities': len([ent for ent in doc.ents if ent.label_ == 'TEST']),
                'anatomical_entities': len([ent for ent in doc.ents if ent.label_ == 'ANATOMICAL']),
                
                # Numerical patterns (ages, measurements, etc.)
                'number_count': len(re.findall(r'\d+', text)),
                'measurement_patterns': len(re.findall(r'\d+\s*(cm|mm|kg|años|mg)', text)),
                
                # Severity indicators
                'severity_terms': sum(1 for word in ['grave', 'severo', 'crítico', 'avanzado', 'agresivo'] 
                                    if word in text.lower()),
                
                # Temporal expressions
                'temporal_terms': sum(1 for word in ['meses', 'años', 'días', 'semanas', 'crónico', 'agudo'] 
                                    if word in text.lower())
            }
            
            features.append(list(text_features.values()))
        
        return np.array(features)

class EnhancedTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Enhanced TF-IDF with medical-specific preprocessing"""
    
    def __init__(self, max_features=1000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
    def fit(self, X, y=None):
        # Medical stop words in Spanish
        medical_stop_words = {
            'paciente', 'presenta', 'refiere', 'años', 'año', 'caso', 'clínico',
            'hospital', 'servicio', 'día', 'días', 'vez', 'veces'
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=list(medical_stop_words),
            lowercase=True,
            token_pattern=r'\b[a-záéíóúñ]{2,}\b',  # Spanish characters
            sublinear_tf=True,
            smooth_idf=True
        )
        
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X):
        return self.vectorizer.transform(X)