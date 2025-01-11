import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

class MetricsCalculator:
    """Class for calculating various GCA metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
        
    def cosine_similarity(self, vec1: List[Tuple[int, float]], 
                         vec2: List[Tuple[int, float]]) -> float:
        """
        Calculate cosine similarity between two sparse vectors.
        
        Args:
            vec1: First sparse vector as list of (index, value) tuples
            vec2: Second sparse vector as list of (index, value) tuples
            
        Returns:
            float: Cosine similarity between the vectors
        """
        # Convert sparse vectors to dictionaries for easier lookup
        dict1 = dict(vec1)
        dict2 = dict(vec2)
        
        # Get all indices
        indices = set(dict1.keys()) | set(dict2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = sum(dict1.get(i, 0) * dict2.get(i, 0) for i in indices)
        mag1 = np.sqrt(sum(v * v for v in dict1.values()))
        mag2 = np.sqrt(sum(v * v for v in dict2.values()))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
            
        return dot_product / (mag1 * mag2)
        
    def compute_similarity_matrix(self, vectors: List[List[Tuple[int, float]]]) -> pd.DataFrame:
        """
        Compute similarity matrix for a list of vectors.
        
        Args:
            vectors: List of sparse vectors
            
        Returns:
            pd.DataFrame: Similarity matrix
        """
        n = len(vectors)
        matrix = pd.DataFrame(0, index=range(1, n+1), columns=range(1, n+1), dtype=float)
        
        for i in range(n):
            for j in range(i, n):
                sim = self.cosine_similarity(vectors[i], vectors[j])
                matrix.loc[i+1, j+1] = sim
                if i != j:
                    matrix.loc[j+1, i+1] = sim
                    
        return matrix
        
    def compute_nct(self, historical_vectors: List[List[Tuple[int, float]]], 
                   current_vector: List[Tuple[int, float]]) -> float:
        """
        Compute Newness Contribution Trend (NCT) for a vector given historical vectors.
        
        Args:
            historical_vectors: List of previous contribution vectors
            current_vector: Current contribution vector
            
        Returns:
            float: NCT score between 0 and 1
        """
        if not historical_vectors:
            return 1.0  # First contribution is maximally new
            
        # Calculate maximum similarity with any historical vector
        similarities = [
            self.cosine_similarity(current_vector, hist_vec)
            for hist_vec in historical_vectors
        ]
        
        # Use maximum similarity (higher means less new)
        max_similarity = max(similarities)
        
        # Convert to newness score (1 - similarity)
        # Scale it to ensure some novelty is maintained
        newness = 1.0 - (0.8 * max_similarity)  # Scale factor of 0.8 ensures minimum novelty of 0.2
        
        return newness

    def get_cosine_similarity_matrix(self, vectors: List[List[Tuple[Any, float]]], time_list: List[int]) -> pd.DataFrame:
        """
        Compute cosine similarity matrix for all vectors.
        
        Args:
            vectors: List of document vectors
            time_list: List of time points
            
        Returns:
            pd.DataFrame: Similarity matrix
        """
        n = len(vectors)
        matrix = pd.DataFrame(0, index=time_list, columns=time_list)
        
        for i in range(n):
            for j in range(n):
                matrix.loc[i, j] = self.cosine_similarity(vectors[i], vectors[j])
                
        return matrix

    def compute_Di(self, di: List[float], ci: List[str]) -> float:
        """
        Compute communication density.
        
        Args:
            di: Document vector
            ci: List of words
            
        Returns:
            float: Communication density value
        """
        word_len = sum(len(t) for t in ci)
        return np.linalg.norm(di) / word_len if word_len > 0 else 0
