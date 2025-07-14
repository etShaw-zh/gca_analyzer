"""
Utility functions for GCA Analyzer.

This module provides utility functions for data processing and analysis.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

from typing import Callable, List, Union

import numpy as np
import pandas as pd

from .logger import logger


def normalize_metrics(
    data: pd.DataFrame, metrics: Union[str, List[str]], inplace: bool = False
) -> pd.DataFrame:
    """
    Normalize metrics in a DataFrame to the range [0, 1] using min-max normalization.

    Args:
        data (pd.DataFrame): Input DataFrame containing metrics.
        metrics (Union[str, List[str]]): Column name(s) of metrics to normalize.
        inplace (bool, optional): Whether to modify the input DataFrame or return a new one.
            Defaults to False.

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


def measure_time(func_name: str) -> Callable:
    """
    Decorator to measure and log execution time of functions.

    Args:
        func_name (str): Name of the function or operation being timed

    Returns:
        Callable: A decorator function that wraps the original function
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func_name} took {elapsed_time:.2f} seconds")
            return result

        return wrapper

    return decorator


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

    if vec1.shape != vec2.shape:  # pragma: no cover
        raise ValueError(
            "Input vectors must have the same number of elements"
        )  # pragma: no cover

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
    current_data: pd.DataFrame,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate cosine similarity matrix for a list of vectors using optimized vectorized
    operations.

    This function now uses sklearn's optimized cosine similarity calculation with batch
    processing for large datasets and optional progress tracking.

    Args:
        vectors: List of vectors as numpy arrays or DataFrame with vector components
        seq_list: List of sequential message numbers
        current_data: DataFrame containing the messages
        batch_size: Size of batches for processing large datasets (default: 1000)
        show_progress: Whether to show progress bar for large operations (default: True)

    Returns:
        pd.DataFrame: Cosine similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    from tqdm import tqdm

    # Input validation
    if len(seq_list) == 0 or current_data.empty:
        logger.warning("Empty input provided to cosine_similarity_matrix")
        return pd.DataFrame()

    # Convert DataFrame to list of vectors if necessary
    if isinstance(vectors, pd.DataFrame):
        vectors = [np.array(vectors.iloc[i]) for i in range(len(vectors))]

    # Validate vectors
    if len(vectors) == 0:  # pragma: no cover
        logger.warning(
            "Empty vectors provided to cosine_similarity_matrix"
        )  # pragma: no cover
        return pd.DataFrame()  # pragma: no cover

    # Check if we have enough vectors for all sequences
    if len(vectors) < len(seq_list):
        logger.error("Not enough vectors for all sequences")
        return pd.DataFrame()

    try:
        # Create mapping from sequence numbers to vector indices
        seq_to_idx = {}
        seq_iter = (
            tqdm(seq_list, desc="Mapping sequences", disable=not show_progress)
            if show_progress
            else seq_list
        )
        for seq in seq_iter:
            matches = current_data[current_data.seq_num == seq]
            if not matches.empty:  # pragma: no cover
                idx = matches.index[0]
                if idx < len(vectors):  # pragma: no cover
                    seq_to_idx[seq] = idx

        # Get valid vectors and their sequence numbers
        valid_seqs = list(seq_to_idx.keys())
        if not valid_seqs:  # pragma: no cover
            logger.error("No valid sequences found")  # pragma: no cover
            return pd.DataFrame()  # pragma: no cover

        valid_vectors = [vectors[seq_to_idx[seq]] for seq in valid_seqs]
        num_vectors = len(valid_vectors)

        # Convert to numpy array for vectorized operations
        vectors_array = np.array(valid_vectors)

        # Handle potential shape inconsistencies
        if vectors_array.ndim > 2:
            vectors_array = vectors_array.reshape(num_vectors, -1)

        # For large datasets, use batch processing to manage memory
        if num_vectors > batch_size:
            logger.info(f"Processing {num_vectors} vectors in batches of {batch_size}")

            # Initialize similarity matrix
            similarity_matrix = np.zeros((num_vectors, num_vectors))

            # Process in batches
            batch_iter = range(0, num_vectors, batch_size)
            if show_progress:
                batch_iter = tqdm(batch_iter, desc="Processing batches", leave=False)

            for i in batch_iter:
                end_i = min(i + batch_size, num_vectors)
                batch_vectors = vectors_array[i:end_i]

                # Calculate similarity for this batch against all vectors
                batch_similarities = sklearn_cosine_similarity(
                    batch_vectors, vectors_array
                )
                similarity_matrix[i:end_i] = batch_similarities
        else:
            # For smaller datasets, process all at once
            if show_progress and num_vectors > 100:
                logger.info("Calculating cosine similarity matrix...")
            similarity_matrix = sklearn_cosine_similarity(vectors_array)

        # Create DataFrame with proper sequence number indexing
        cosine_matrix = pd.DataFrame(
            similarity_matrix, index=valid_seqs, columns=valid_seqs, dtype=float
        )

        # Fill in missing sequences with zeros if needed
        if len(valid_seqs) < len(seq_list):
            missing_seqs = [seq for seq in seq_list if seq not in valid_seqs]
            for seq in missing_seqs:
                cosine_matrix.loc[seq] = 0.0
                cosine_matrix[seq] = 0.0

            # Reorder to match original seq_list order
            cosine_matrix = cosine_matrix.reindex(
                index=seq_list, columns=seq_list, fill_value=0.0
            )

    except Exception as e:  # pragma: no cover
        logger.error(
            f"Error calculating similarity matrix: {str(e)}"
        )  # pragma: no cover
        return pd.DataFrame()  # pragma: no cover

    return cosine_matrix
