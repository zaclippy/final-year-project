from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalFeatureSelector:
    """Comprehensive feature selection for medical text classification"""
    
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.chi2_selector = None
        self.mutual_info_selector = None
        self.rfe_selector = None
        
    def fit_chi2_selection(self, X, y):
        """Chi-squared feature selection"""
        self.chi2_selector = SelectKBest(chi2, k=self.n_features)
        self.chi2_selector.fit(X, y)
        return self
    
    def fit_mutual_info_selection(self, X, y):
        """Mutual information feature selection"""
        self.mutual_info_selector = SelectKBest(mutual_info_classif, k=self.n_features)
        self.mutual_info_selector.fit(X, y)
        return self
    
    def fit_rfe_selection(self, X, y):
        """Recursive feature elimination with Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rfe_selector = RFE(rf, n_features_to_select=self.n_features)
        self.rfe_selector.fit(X, y)
        return self
    
    def transform_chi2(self, X):
        return self.chi2_selector.transform(X)
    
    def transform_mutual_info(self, X):
        return self.mutual_info_selector.transform(X)
    
    def transform_rfe(self, X):
        return self.rfe_selector.transform(X)
    
    def plot_feature_importance(self, feature_names):
        """Visualize feature importance scores"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Chi-squared scores
        if self.chi2_selector:
            chi2_scores = self.chi2_selector.scores_
            top_indices = np.argsort(chi2_scores)[-20:]
            axes[0].barh(range(20), chi2_scores[top_indices])
            axes[0].set_title('Top 20 Chi-squared Scores')
            axes[0].set_xlabel('Chi-squared Score')
        
        # Mutual information scores  
        if self.mutual_info_selector:
            mi_scores = self.mutual_info_selector.scores_
            top_indices = np.argsort(mi_scores)[-20:]
            axes[1].barh(range(20), mi_scores[top_indices])
            axes[1].set_title('Top 20 Mutual Information Scores')
            axes[1].set_xlabel('Mutual Information Score')
        
        # RFE ranking
        if self.rfe_selector:
            rfe_ranking = self.rfe_selector.ranking_
            selected_features = np.where(rfe_ranking == 1)[0][:20]
            axes[2].barh(range(len(selected_features)), [1] * len(selected_features))
            axes[2].set_title('RFE Selected Features')
            axes[2].set_xlabel('Selected (1) or Not (0)')
        
        plt.tight_layout()
        plt.show()