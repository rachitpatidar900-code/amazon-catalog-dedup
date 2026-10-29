import re
import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.normalizer import AttributeNormalizer, NormalizedAttributes


class MatchDecision(str, Enum):
    EXACT_DUPLICATE = "EXACT_DUPLICATE"      # Auto-merge candidate (same item, redundant listing)
    POTENTIAL_VARIATION = "POTENTIAL_VARIATION"  # Parent-child relation (e.g., different color/size)
    DISTINCT = "DISTINCT"                    # Standalone different product


class BaselineMatcher:
    """
    Empirical Baseline: TF-IDF n-gram vectorizer with cosine similarity.
    Provides sub-2ms matching as a benchmark before heavy neural embeddings.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1
        )
        self.is_fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        if not self.is_fitted:
            self.fit([text_a, text_b])
        tfidf_matrices = self.vectorizer.transform([text_a, text_b])
        sim = cosine_similarity(tfidf_matrices[0], tfidf_matrices[1])[0][0]
        return float(sim)


class SemanticMatcher:
    """
    Production Matcher: Uses SentenceTransformer bi-encoder representations
    combined with structured attribute divergence checks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.normalizer = AttributeNormalizer()

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                # Graceful fallback for lightweight testing environments
                self._model = "FALLBACK_TFIDF"
        return self._model

    def get_embedding(self, text: str) -> np.ndarray:
        if self.model == "FALLBACK_TFIDF":
            # Deterministic character & word n-gram projection when sentence_transformers is not installed
            from sklearn.feature_extraction.text import HashingVectorizer
            hv = HashingVectorizer(n_features=384, alternate_sign=False, ngram_range=(1, 3))
            sparse_vec = hv.transform([text.lower()]).toarray()[0]
            norm = np.linalg.norm(sparse_vec)
            if norm == 0:
                return np.zeros(384)
            return sparse_vec / norm
        return self.model.encode(text, normalize_embeddings=True)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)
        return float(np.dot(emb_a, emb_b))

    def classify_pair(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
        sim_threshold_duplicate: float = 0.85,
        sim_threshold_variation: float = 0.65
    ) -> Tuple[MatchDecision, float, Dict[str, Any]]:
        """
        Classifies product relationship based on semantic similarity + attribute delta.
        """
        title_a = item_a.get("title", "")
        title_b = item_b.get("title", "")

        similarity = self.compute_similarity(title_a, title_b)

        # Extract normalized attributes to detect variations
        attr_a = self.normalizer.normalize(item_a.get("raw_attributes", {}), title=title_a)
        attr_b = self.normalizer.normalize(item_b.get("raw_attributes", {}), title=title_b)

        # Detect variation signals (e.g. different colors or storage)
        varying_features = []
        if attr_a.color and attr_b.color and attr_a.color != attr_b.color:
            varying_features.append(f"color: {attr_a.color} vs {attr_b.color}")

        if attr_a.storage_gb and attr_b.storage_gb and attr_a.storage_gb != attr_b.storage_gb:
            varying_features.append(f"storage: {attr_a.storage_gb}GB vs {attr_b.storage_gb}GB")

        if attr_a.volume_ml and attr_b.volume_ml and abs(attr_a.volume_ml - attr_b.volume_ml) > 50:
            varying_features.append(f"volume: {attr_a.volume_ml}ml vs {attr_b.volume_ml}ml")

        # Decision Logic
        if similarity >= sim_threshold_duplicate and not varying_features:
            decision = MatchDecision.EXACT_DUPLICATE
        elif similarity >= sim_threshold_variation or (similarity >= sim_threshold_duplicate and varying_features):
            decision = MatchDecision.POTENTIAL_VARIATION
        else:
            decision = MatchDecision.DISTINCT

        metadata = {
            "similarity_score": round(similarity, 4),
            "varying_features": varying_features,
            "normalized_a": attr_a.model_dump(exclude_none=True),
            "normalized_b": attr_b.model_dump(exclude_none=True)
        }

        return decision, similarity, metadata
