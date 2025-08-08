import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import pickle
import hashlib
import glob

class OptimizedFeatureEngine:

    def load_cache(self):
        """Load cached features from disk without recomputation."""
        cache_file = self.cache_path or 'feature_cache.pkl'
        import glob
        base, ext = os.path.splitext(cache_file)
        pattern = f"{base}_*.pkl"
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No cached features found matching {pattern}")
        cache_file = max(candidates, key=os.path.getmtime)
        with open(cache_file, 'rb') as f:
            obj = pickle.load(f)
        features = obj.get('features', None)
        if features is None:
            raise RuntimeError(f"Cached features file {cache_file} is missing 'features' key.")
        return features
    """
    Enhanced feature generator for RL trading environments with improved compression and selection.
    - Uses RobustScaler for better handling of outliers in financial data
    - Combines statistical selection with PCA for optimal feature retention
    - Implements adaptive component selection based on explained variance
    - Maintains more components to preserve trading-relevant patterns
    """

    def __init__(self, 
                 target_variance: float = 0.95, 
                 min_components: int = 20,
                 max_components: int = 50,
                 variance_threshold: float = 1e-6, 
                 cache_path: str = None,
                 reuse_existing: bool = True,
                 keep_last: int = 3):
        self.target_variance = target_variance
        self.min_components = min_components
        self.max_components = max_components
        self.variance_threshold = variance_threshold
        self.cache_path = cache_path
        self.reuse_existing = reuse_existing
        self.keep_last = keep_last

        self.var_thresh = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.n_components_used = min_components

    def _determine_optimal_components(self, X: np.ndarray) -> int:
        """Determine optimal number of PCA components based on explained variance"""
        temp_pca = PCA(svd_solver='full')
        temp_pca.fit(X)
        
        cumsum_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        
        optimal_components = np.argmax(cumsum_variance >= self.target_variance) + 1
        
        optimal_components = max(self.min_components, 
                               min(optimal_components, self.max_components))
        
        return optimal_components

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Enhanced fitting with adaptive component selection and feature importance
        
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,) - target for feature selection (optional)
        """
        cache_file = self.cache_path or 'feature_cache.pkl'
        base, ext = os.path.splitext(cache_file)
        if self.reuse_existing:
            features, obj, path = self._find_valid_cached_features(base, X.shape)
            if features is not None:
                self.var_thresh = obj['var_thresh']
                self.scaler = obj['scaler']
                self.feature_selector = obj.get('feature_selector')
                self.pca = obj['pca']
                self.n_components_used = obj['n_components_used']
                print(f"✅ Loaded cached features: {features.shape} from {os.path.basename(path)}")
                return features

        cache_id = self._stable_cache_id(X.shape)
        cache_file = f"{base}_{cache_id}.pkl"

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    obj = pickle.load(f)
                features = obj.get('features', None)
                if (features is not None and hasattr(features, 'shape') and features.shape[0] == X.shape[0] and 
                    obj.get('target_variance', 0) >= self.target_variance * 0.95):
                    self.var_thresh = obj['var_thresh']
                    self.scaler = obj['scaler']
                    self.feature_selector = obj.get('feature_selector')
                    self.pca = obj['pca']
                    self.n_components_used = obj['n_components_used']
                    print(f"✅ Loaded deterministic cache: {features.shape}")
                    return features
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}")

        print("🔄 Computing enhanced features...")
        
        self.var_thresh = VarianceThreshold(self.variance_threshold)
        X_var = self.var_thresh.fit_transform(X)
        print(f"📊 After variance threshold: {X_var.shape[1]} features")
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_var)
        
        if y is not None and len(np.unique(y)) > 1:
            n_select = min(X_scaled.shape[1], max(100, X_scaled.shape[1] // 2))
            self.feature_selector = SelectKBest(f_regression, k=n_select)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            print(f"📊 After feature selection: {X_selected.shape[1]} features")
        else:
            X_selected = X_scaled
            print("📊 No target provided, skipping feature selection")

        self.n_components_used = self._determine_optimal_components(X_selected)
        self.pca = PCA(n_components=self.n_components_used, svd_solver='full')
        X_pca = self.pca.fit_transform(X_selected)
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"📊 PCA: {self.n_components_used} components, {explained_var:.3f} variance explained")
        
        enhanced_features = self._add_regime_features(X_pca, X)
        
        cache_data = {
            'features': enhanced_features,
            'var_thresh': self.var_thresh,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'n_components_used': self.n_components_used,
            'target_variance': explained_var,
            'original_shape': X.shape
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cached features to: {cache_file}")
            self._cleanup_old_caches(base)
        except Exception as e:
            print(f"⚠️ Caching failed: {e}")

        return enhanced_features

    def _add_regime_features(self, X_pca: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """Add market regime and stability indicators"""
        n_samples = X_pca.shape[0]
        
        window = min(20, n_samples // 4)
        if window > 1:
            vol_regime = np.zeros(n_samples)
            for i in range(window, n_samples):
                vol_regime[i] = np.std(X_pca[i-window:i, 0])
            vol_regime[:window] = vol_regime[window]
            vol_regime = (vol_regime - np.mean(vol_regime)) / (np.std(vol_regime) + 1e-8)
        else:
            vol_regime = np.zeros(n_samples)
            
        trend_persistence = np.zeros(n_samples)
        if n_samples > 10:
            for i in range(10, n_samples):
                recent_pcs = X_pca[max(0, i-10):i, :3]
                if recent_pcs.shape[0] > 5:
                    autocorr = np.mean([
                        np.corrcoef(recent_pcs[:-1, j], recent_pcs[1:, j])[0, 1]
                        for j in range(min(3, recent_pcs.shape[1]))
                        if not np.isnan(np.corrcoef(recent_pcs[:-1, j], recent_pcs[1:, j])[0, 1])
                    ])
                    trend_persistence[i] = autocorr if not np.isnan(autocorr) else 0
        
        stability = np.zeros(n_samples)
        if n_samples > 5:
            for i in range(5, n_samples):
                recent = X_pca[max(0, i-5):i]
                if recent.shape[0] > 2:
                    mean_pattern = np.mean(recent, axis=0)
                    current = X_pca[i]
                    corr = np.corrcoef(mean_pattern, current)[0, 1]
                    stability[i] = corr if not np.isnan(corr) else 0
        
        regime_features = np.column_stack([
            vol_regime.reshape(-1, 1),
            trend_persistence.reshape(-1, 1), 
            stability.reshape(-1, 1)
        ])
        
        return np.hstack([X_pca, regime_features])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted pipeline"""
        if self.var_thresh is None or self.scaler is None or self.pca is None:
            raise RuntimeError("You must call fit_transform before transform.")

        X_var = self.var_thresh.transform(X)
        X_scaled = self.scaler.transform(X_var)
        
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
            
        X_pca = self.pca.transform(X_selected)
        
        enhanced_features = self._add_regime_features(X_pca, X)
        
        return enhanced_features

    def get_feature_importance(self) -> dict:
        """Get feature importance metrics"""
        if self.pca is None:
            return {}
            
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'n_components': self.n_components_used,
            'total_explained_variance': np.sum(self.pca.explained_variance_ratio_),
            'feature_selection_used': self.feature_selector is not None
        }

    def _stable_cache_id(self, X_shape) -> str:
        """Create a deterministic cache id based on data shape and params."""
        key = f"shape={X_shape}|tv={self.target_variance}|min={self.min_components}|max={self.max_components}|vt={self.variance_threshold}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

    def _load_cache_file(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj.get('features', None), obj

    def _find_valid_cached_features(self, base: str, X_shape) -> tuple:
        """Return (features, obj, path) for the newest valid cache matching X_shape and params."""
        pattern = f"{base}_*.pkl"
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for p in candidates:
            try:
                features, obj = self._load_cache_file(p)
                if features is None:
                    continue
                orig_shape = obj.get('original_shape')
                tv = obj.get('target_variance', 0)
                if orig_shape == X_shape and tv >= self.target_variance * 0.95:
                    return features, obj, p
            except Exception:
                continue
        return None, None, None

    def _cleanup_old_caches(self, base: str):
        if self.keep_last is None or self.keep_last <= 0:
            return
        pattern = f"{base}_*.pkl"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for p in files[self.keep_last:]:
            try:
                os.remove(p)
            except Exception:
                pass
