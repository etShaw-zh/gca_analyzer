"""Test module for utility functions."""
import numpy as np
import pandas as pd
import pytest

from gca_analyzer.utils import (
    normalize_metrics,
    cosine_similarity,
    cosine_similarity_matrix
)


def test_normalize_metrics_single_column():
    """Test normalizing a single metric column."""
    data = pd.DataFrame({
        'metric1': [1, 2, 3, 4, 5],
        'metric2': [10, 20, 30, 40, 50]
    })
    result = normalize_metrics(data, 'metric1')
    expected = pd.DataFrame({
        'metric1': [0.0, 0.25, 0.5, 0.75, 1.0],
        'metric2': [10, 20, 30, 40, 50]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_normalize_metrics_multiple_columns():
    """Test normalizing multiple metric columns."""
    data = pd.DataFrame({
        'metric1': [1, 2, 3, 4, 5],
        'metric2': [10, 20, 30, 40, 50]
    })
    result = normalize_metrics(data, ['metric1', 'metric2'])
    expected = pd.DataFrame({
        'metric1': [0.0, 0.25, 0.5, 0.75, 1.0],
        'metric2': [0.0, 0.25, 0.5, 0.75, 1.0]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_normalize_metrics_inplace():
    """Test normalizing metrics inplace."""
    data = pd.DataFrame({
        'metric1': [1, 2, 3, 4, 5]
    })
    original_id = id(data)
    result = normalize_metrics(data, 'metric1', inplace=True)
    assert id(result) == original_id  # Should modify in place


def test_normalize_metrics_same_values():
    """Test normalizing metrics when all values are the same."""
    data = pd.DataFrame({
        'metric1': [5, 5, 5, 5, 5]
    })
    result = normalize_metrics(data, 'metric1')
    expected = pd.DataFrame({
        'metric1': [0, 0, 0, 0, 0]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_cosine_similarity_basic():
    """Test basic cosine similarity calculation."""
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    result = cosine_similarity(vec1, vec2)
    expected = 0.9746318461970762
    assert np.isclose(result, expected)


def test_cosine_similarity_zero_vector():
    """Test cosine similarity with zero vector."""
    vec1 = np.array([0, 0, 0])
    vec2 = np.array([1, 2, 3])
    result = cosine_similarity(vec1, vec2)
    assert result == 0.0


def test_cosine_similarity_2d_vectors():
    """Test cosine similarity with 2D vectors."""
    vec1 = np.array([[1, 2], [3, 4]])
    vec2 = np.array([[5, 6], [7, 8]])
    result = cosine_similarity(vec1, vec2)
    # The expected value should be the cosine similarity of flattened vectors [1,2,3,4] and [5,6,7,8]
    expected = 0.9688639316269662
    assert np.isclose(result, expected)


def test_cosine_similarity_matrix_basic():
    """Test basic cosine similarity matrix calculation."""
    vectors = pd.DataFrame({
        'dim1': [0.1, 0.2, 0.3],
        'dim2': [0.4, 0.5, 0.6]
    })
    seq_list = [1, 2, 3]
    current_data = pd.DataFrame({
        'seq_num': [1, 2, 3],
        'text': ['a', 'b', 'c']
    })
    result = cosine_similarity_matrix(vectors, seq_list, current_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert (result.values >= 0).all() and (result.values <= 1).all()


def test_cosine_similarity_matrix_empty_input():
    """Test cosine similarity matrix with empty input."""
    vectors = pd.DataFrame()
    seq_list = []
    current_data = pd.DataFrame()
    result = cosine_similarity_matrix(vectors, seq_list, current_data)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_cosine_similarity_matrix_error_handling():
    """Test error handling in cosine similarity matrix calculation."""
    vectors = pd.DataFrame({
        'dim1': [0.1, 0.2],
        'dim2': [0.4, 0.5]
    })
    seq_list = [1, 2, 3]  # More sequences than vectors
    current_data = pd.DataFrame({
        'seq_num': [1, 2, 3],
        'text': ['a', 'b', 'c']
    })
    result = cosine_similarity_matrix(vectors, seq_list, current_data)
    assert isinstance(result, pd.DataFrame)
    assert result.empty  # Should return empty DataFrame on error
