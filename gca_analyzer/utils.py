"""
Utility functions for GCA Analyzer.

This module provides utility functions for data processing and analysis.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import pandas as pd
import numpy as np
from typing import List, Union

def normalize_metrics(data: pd.DataFrame, metrics: Union[str, List[str]], inplace: bool = False) -> pd.DataFrame:
    """
    Normalize metrics in a DataFrame to the range [0, 1] using min-max normalization.

    Args:
        data (pd.DataFrame): Input DataFrame containing metrics.
        metrics (Union[str, List[str]]): Column name(s) of metrics to normalize.
        inplace (bool, optional): Whether to modify the input DataFrame or return a new one. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with normalized metrics.
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    if not inplace:
        data = data.copy()

    for metric in metrics:
        min_val = data[metric].min()
        max_val = data[metric].max()
        if max_val != min_val:  # Avoid division by zero
            data[metric] = (data[metric] - min_val) / (max_val - min_val)
        else:
            data[metric] = 0  # If all values are the same, set to 0

    return data

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two numpy arrays.
    
    Args:
        vec1: First vector (numpy array)
        vec2: Second vector (numpy array)
        
    Returns:
        float: Cosine similarity between the vectors
    """
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    norm1 = np.linalg.norm(vec1_flat)
    norm2 = np.linalg.norm(vec2_flat)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1_flat, vec2_flat) / (norm1 * norm2))

def cosine_similarity_matrix(vectors: List[np.ndarray], seq_list: List[int], current_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cosine similarity matrix for a list of vectors.
    
    Args:
        vectors: List of vectors as numpy arrays
        seq_list: List of sequential message numbers
        current_data: DataFrame containing the messages
        
    Returns:
        pd.DataFrame: Cosine similarity matrix
    """
    if not vectors or not seq_list or current_data.empty:
        logger.warning("Empty input provided to cosine_similarity_matrix")
        return pd.DataFrame()

    cosine_matrix = pd.DataFrame(0.0, index=seq_list, columns=seq_list, dtype=float)
    
    seq_to_idx = {
        seq: current_data[current_data.seq_num == seq].index[0]
        for seq in seq_list
        if not current_data[current_data.seq_num == seq].empty
    }
    
    try:
        valid_vectors = np.array([vectors[seq_to_idx[seq]] for seq in seq_list 
                                if seq in seq_to_idx and seq_to_idx[seq] < len(vectors)])
        
        if len(valid_vectors) > 0:
            norms = np.linalg.norm(valid_vectors, axis=1, keepdims=True)
            normalized_vectors = np.divide(valid_vectors, norms, where=norms!=0)
            
            similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
            
            valid_seq_list = [seq for seq in seq_list if seq in seq_to_idx 
                            and seq_to_idx[seq] < len(vectors)]
            for i, seq_i in enumerate(valid_seq_list):
                for j, seq_j in enumerate(valid_seq_list):
                    if i < j:  # Only fill upper triangle and mirror
                        similarity = float(similarity_matrix[i, j])
                        cosine_matrix.loc[seq_i, seq_j] = similarity
                        cosine_matrix.loc[seq_j, seq_i] = similarity
    except Exception as e:
        logger.error(f"Error calculating similarity matrix: {str(e)}")
        
    return cosine_matrix