"""
Utility functions for GCA Analyzer.

This module provides utility functions for data processing and analysis.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

from typing import List, Union

import numpy as np
import pandas as pd

from .logger import logger


def normalize_metrics(
    data: pd.DataFrame,
    metrics: Union[str, List[str]],
    inplace: bool = False
) -> pd.DataFrame:
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
    # Ensure arrays are float type for numerical stability
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)

    # Reshape arrays to 1D if needed
    if vec1.ndim > 1:
        vec1 = vec1.reshape(-1)
    if vec2.ndim > 1:
        vec2 = vec2.reshape(-1)

    if vec1.shape != vec2.shape:
        raise ValueError("Input vectors must have the same number of elements")

    # Calculate norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Calculate cosine similarity with improved numerical stability
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)

    # Handle numerical errors that might make similarity slightly outside [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)

    return float(similarity)


def cosine_similarity_matrix(
    vectors: Union[List[np.ndarray], pd.DataFrame],
    seq_list: List[int],
    current_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate cosine similarity matrix for a list of vectors.

    Args:
        vectors: List of vectors as numpy arrays or DataFrame with vector components
        seq_list: List of sequential message numbers
        current_data: DataFrame containing the messages

    Returns:
        pd.DataFrame: Cosine similarity matrix
    """
    # Input validation
    if len(seq_list) == 0 or current_data.empty:
        logger.warning("Empty input provided to cosine_similarity_matrix")
        return pd.DataFrame()

    # Convert DataFrame to list of vectors if necessary
    if isinstance(vectors, pd.DataFrame):
        vectors = [np.array(vectors.iloc[i]) for i in range(len(vectors))]

    # Validate vectors
    if len(vectors) == 0:
        logger.warning("Empty vectors provided to cosine_similarity_matrix")
        return pd.DataFrame()

    # Check if we have enough vectors for all sequences
    if len(vectors) < len(seq_list):
        logger.error("Not enough vectors for all sequences")
        return pd.DataFrame()

    # Initialize similarity matrix with zeros
    cosine_matrix = pd.DataFrame(0.0, index=seq_list, columns=seq_list, dtype=float)

    try:
        # Create mapping from sequence numbers to vector indices
        seq_to_idx = {}
        for seq in seq_list:
            matches = current_data[current_data.seq_num == seq]
            if not matches.empty:
                idx = matches.index[0]
                if idx < len(vectors):
                    seq_to_idx[seq] = idx

        # Get valid vectors and their sequence numbers
        valid_seqs = list(seq_to_idx.keys())
        if not valid_seqs:
            logger.error("No valid sequences found")
            return pd.DataFrame()

        valid_vectors = [vectors[seq_to_idx[seq]] for seq in valid_seqs]

        # Calculate similarities for valid vectors
        for i, seq_i in enumerate(valid_seqs):
            for j, seq_j in enumerate(valid_seqs):
                if i <= j:  # Calculate upper triangle and diagonal
                    try:
                        similarity = cosine_similarity(valid_vectors[i], valid_vectors[j])
                        cosine_matrix.loc[seq_i, seq_j] = similarity
                        if i != j:  # Mirror for lower triangle
                            cosine_matrix.loc[seq_j, seq_i] = similarity
                    except Exception as e:
                        logger.error(f"Error calculating similarity for vectors {i} and {j}: {str(e)}")
                        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error calculating similarity matrix: {str(e)}")
        return pd.DataFrame()

    return cosine_matrix