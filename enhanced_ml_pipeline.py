from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from enhanced_feature_engineering import EnhancedTfidfVectorizer
from feature_selection import MedicalFeatureExtractor

class EnhancedMedicalClassifier:
    """Enhanced classification pipeline for medical text"""
    
    def __init__(self, nlp_model=None):
        self.nlp_model = nlp_model
        self.pipeline = None
        self.best_model = None
        
    def create_pipeline(self):
        """Create enhanced feature extraction and classification pipeline"""
        
        # Feature extraction pipeline
        feature_pipeline = FeatureUnion([
            # Enhanced TF-IDF features
            ('tfidf', Pipeline([
                ('tfidf_vectorizer', EnhancedTfidfVectorizer(max_features=800, ngram_range=(1, 3))),
                ('feature_selection', SelectKBest(chi2, k=400))
            ])),
            
            # Medical domain features
            ('medical_features', Pipeline([
                ('medical_extractor', MedicalFeatureExtractor(self.nlp_model)),
                ('scaler', StandardScaler())
            ]))
        ])
        
        # Full pipeline with classification
        self.pipeline = Pipeline([
            ('features', feature_pipeline),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        return self.pipeline
    
    def grid_search_optimization(self, X_train, y_train):
        """Comprehensive grid search for best parameters"""
        
        # Define parameter grids for different classifiers
        param_grids = [
            {
                'classifier': [LogisticRegression(random_state=42)],
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear'],
                'features__tfidf__feature_selection__k': [200, 400, 600]
            },
            {
                'classifier': [RandomForestClassifier(random_state=42)],
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'features__tfidf__feature_selection__k': [200, 400, 600]
            },
            {
                'classifier': [GradientBoostingClassifier(random_state=42)],
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'features__tfidf__feature_selection__k': [200, 400, 600]
            }
        ]
        
        best_score = 0
        best_pipeline = None
        
        for param_grid in param_grids:
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=5, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_pipeline = grid_search.best_estimator_
        
        self.best_model = best_pipeline
        return best_pipeline, best_score
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        if not self.best_model:
            raise ValueError("Model not trained yet. Run grid_search_optimization first.")
        
        y_pred = self.best_model.predict(X_test)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.best_model, X_test, y_test, cv=5, scoring='f1_weighted')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'cv_scores': cv_scores
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.best_model, filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        self.best_model = joblib.load(filepath)