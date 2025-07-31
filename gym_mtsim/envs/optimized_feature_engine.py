import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import os
import pickle

class OptimizedFeatureEngine:
    """
    Optimized feature generator for RL trading environments.
    - Removes near-constant features using VarianceThreshold.
    - Standardizes features for PCA stability.
    - Compresses features using PCA to a small latent space (default: 10 components).
    - Uses caching for faster re-runs.

    This reduces 30k+ features down to a tiny compressed state space 
    while retaining high variance.
    """

    def __init__(self, n_pca_components: int = 10, variance_threshold: float = 1e-5, cache_path: str = None):
        self.n_pca_components = n_pca_components
        self.variance_threshold = variance_threshold
        self.cache_path = cache_path

        self.var_thresh = None
        self.scaler = None
        self.pca = None


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits VarianceThreshold, StandardScaler, and PCA on the given feature matrix,
        then returns the transformed feature matrix.

        X: np.ndarray of shape (n_samples, n_features)
        """
        cache_file = self.cache_path or 'feature_cache.pkl'
        cache_id = f"{X.shape}_{self.n_pca_components}_{self.variance_threshold}"
        cache_file = cache_file.replace('.pkl', f'_{abs(hash(cache_id))}.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                obj = pickle.load(f)
            features = obj['features']
            if features.shape[0] == X.shape[0] and features.shape[1] == self.n_pca_components:
                self.var_thresh = obj['var_thresh']
                self.scaler = obj['scaler']
                self.pca = obj['pca']
                return features

        self.var_thresh = VarianceThreshold(self.variance_threshold)
        X_var = self.var_thresh.fit_transform(X)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_var)

        self.pca = PCA(n_components=self.n_pca_components, svd_solver='full')
        X_pca = self.pca.fit_transform(X_scaled)

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'features': X_pca,
                'var_thresh': self.var_thresh,
                'scaler': self.scaler,
                'pca': self.pca
            }, f)

        return X_pca

    def load_features(self) -> np.ndarray:
        """
        Loads features from cache file, if valid. Returns None if not valid.
        """
        cache_file = self.cache_path or 'feature_cache.pkl'
        if not os.path.exists(cache_file):
            base, ext = os.path.splitext(cache_file)
            for fname in os.listdir(os.path.dirname(cache_file) or '.'):
                if fname.startswith(os.path.basename(base)) and fname.endswith(ext):
                    cache_file = os.path.join(os.path.dirname(cache_file), fname)
                    break
            else:
                return None
        with open(cache_file, 'rb') as f:
            obj = pickle.load(f)
        self.var_thresh = obj['var_thresh']
        self.scaler = obj['scaler']
        self.pca = obj['pca']
        return obj['features']

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms new data using the fitted pipeline (VarianceThreshold -> StandardScaler -> PCA).
        """
        if self.var_thresh is None or self.scaler is None or self.pca is None:
            raise RuntimeError("You must call fit_transform before transform.")

        X_var = self.var_thresh.transform(X)
        X_scaled = self.scaler.transform(X_var)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
