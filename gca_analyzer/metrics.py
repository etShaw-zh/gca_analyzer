import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Union
from .logger import logger

class MetricsCalculator:
    """Class for calculating various GCA metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        logger.info("Initializing Metrics Calculator")
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
        Compute Newness Contribution Trend (NCT) using orthogonalization.
        
        Args:
            historical_vectors: List of previous contribution vectors
            current_vector: Current contribution vector
            
        Returns:
            float: NCT score between 0 and 1
        """
        if not historical_vectors:
            return 1.0  # First contribution is maximally new
            
        # Convert sparse vectors to dense format
        def to_dense(vec, max_dim):
            dense = np.zeros(max_dim)
            for idx, val in vec:
                if idx < max_dim:
                    dense[idx] = val
            return dense
            
        # Get maximum dimension
        max_dim = max(
            max(idx for vec in historical_vectors for idx, _ in vec),
            max(idx for idx, _ in current_vector)
        ) + 1
        
        # Convert to dense matrices
        G = np.array([to_dense(vec, max_dim) for vec in historical_vectors])
        d = to_dense(current_vector, max_dim)
        
        # Perform QR decomposition for orthogonalization
        Q, R = np.linalg.qr(G.T)
        
        # Project current vector onto orthogonal space
        d_ortho = d - Q @ (Q.T @ d)
        
        # Compute NCT as ratio of orthogonal component
        nct = np.linalg.norm(d_ortho) / (np.linalg.norm(d_ortho) + np.linalg.norm(Q.T @ d))
        
        return nct

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

    def compute_Di(self, di: List[float], ci: List[str], time_window: int) -> float:
        """
        Compute communication density with temporal weighting.
        
        Args:
            di: Document vector
            ci: List of words
            time_window: Current time window
            
        Returns:
            float: Communication density value
        """
        word_len = sum(len(t) for t in ci)
        if word_len == 0:
            return 0
            
        # Add temporal decay factor
        temporal_weight = np.exp(-0.1 * time_window)  # Exponential decay
        vector_norm = np.linalg.norm(di)
        
        return (vector_norm * temporal_weight) / word_len

    def compute_responsivity_metrics(self, interaction_matrix: pd.DataFrame, 
                                   similarity_matrix: pd.DataFrame,
                                   window_size: int) -> Dict[str, float]:
        """
        Compute responsivity metrics including internal cohesion and social impact.
        
        Args:
            interaction_matrix: Matrix of interactions
            similarity_matrix: Matrix of content similarities
            window_size: Size of sliding window
            
        Returns:
            Dict containing metrics
        """
        n_participants = len(interaction_matrix)
        metrics = {
            'internal_cohesion': 0.0,
            'overall_responsivity': 0.0,
            'social_impact': 0.0
        }
        
        # Calculate weighted interaction strengths
        for i in range(n_participants):
            for j in range(n_participants):
                interaction_strength = 0
                for t in range(window_size-1):
                    # Calculate temporal interaction strength
                    if i == j:  # Internal cohesion
                        interaction_strength += (
                            interaction_matrix.iloc[i,t] * 
                            interaction_matrix.iloc[i,t+1] * 
                            similarity_matrix.iloc[t,t+1]
                        )
                    else:  # Cross-participant interactions
                        interaction_strength += (
                            interaction_matrix.iloc[i,t] * 
                            interaction_matrix.iloc[j,t+1] * 
                            similarity_matrix.iloc[t,t+1]
                        )
                
                # Normalize by window size
                interaction_strength /= (window_size - 1)
                
                if i == j:
                    metrics['internal_cohesion'] += interaction_strength
                else:
                    metrics['overall_responsivity'] += interaction_strength
                    metrics['social_impact'] += interaction_strength
        
        # Normalize metrics
        total_interactions = interaction_matrix.sum().sum()
        if total_interactions > 0:
            metrics['overall_responsivity'] /= total_interactions
            metrics['social_impact'] /= total_interactions
        
        if n_participants > 0:
            metrics['internal_cohesion'] /= n_participants
            
        return metrics

    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all metrics for the given data."""
        logger.info("Starting metrics calculation")
        metrics = {}
        try:
            # Calculate various metrics
            metrics.update(self._calculate_basic_metrics(data))
            metrics.update(self._calculate_advanced_metrics(data))
            logger.debug(f"Calculated metrics: {list(metrics.keys())}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        # TO DO: implement basic metrics calculation
        pass

    def _calculate_advanced_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        # TO DO: implement advanced metrics calculation
        pass
