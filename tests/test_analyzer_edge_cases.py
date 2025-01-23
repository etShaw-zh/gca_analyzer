import pytest
import pandas as pd
import numpy as np
from gca_analyzer.analyzer import GCAAnalyzer
from gca_analyzer.llm_processor import LLMTextProcessor

@pytest.fixture
def analyzer():
    return GCAAnalyzer()

def test_find_best_window_size_no_valid_window(analyzer):
    """Test find_best_window_size when no valid window is found (lines 201-205)"""
    data = pd.DataFrame({
        'person_id': ['A', 'B'] * 5,
        'text': ['Hello'] * 10,
        'time': pd.date_range(start='2024-01-01', periods=10, freq='h'),
        'seq': range(10)
    })
    
    # Set a high threshold that cannot be met
    best_window_size = analyzer.find_best_window_size(
        data, 
        best_window_indices=0.99,  # Very high threshold
        min_num=2,
        max_num=4
    )
    assert best_window_size == 4  # Should return max_num when no valid window found

def test_calculate_personal_given_new_averages_empty_dict(analyzer):
    """Test calculate_personal_given_new_averages with empty dictionaries (lines 462-463)"""
    person_list = ['A', 'B']
    n_c_t_dict = {'A': [0.5], 'B': []}  # One person has data, other empty
    D_i_dict = {'A': [0.3], 'B': []}    # One person has data, other empty
    
    newness, density = analyzer.calculate_personal_given_new_averages(
        person_list, n_c_t_dict, D_i_dict
    )
    
    assert newness['A'] == 0.5
    assert newness['B'] == 0.0
    assert density['A'] == 0.3
    assert density['B'] == 0.0

def test_calculate_batch_lsa_metrics_first_message(analyzer):
    """Test _calculate_batch_lsa_metrics for first message (lines 532-542)"""
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    texts = ['Hello', 'World']
    
    # Test first message (should have newness = 1.0)
    results = analyzer._calculate_batch_lsa_metrics(vectors, texts, 0, 1)
    assert len(results) == 1
    assert results[0][0] == 1.0  # First message should be entirely new

def test_calculate_communication_density_edge_cases(analyzer):
    """Test _calculate_communication_density edge cases (lines 609, 613)"""
    # Test empty text
    density = analyzer._calculate_communication_density(np.array([1.0, 0.0]), '')
    assert density == 0.0
    
    # Test pandas Series input
    vector = pd.Series([1.0, 0.0])
    density = analyzer._calculate_communication_density(vector, 'test')
    assert density > 0.0

def test_calculate_descriptive_statistics_edge_cases(analyzer):
    """Test calculate_descriptive_statistics edge cases (line 803)"""
    metrics = {
        '1': pd.DataFrame({
            'Newness': [0.0, 0.0],  # Mean = 0
            'Communication_density': [1.0, -1.0]  # Mean = 0 but SD > 0
        })
    }
    
    stats = analyzer.calculate_descriptive_statistics(metrics)
    # Access by index since DataFrame is transposed
    assert stats.loc['Newness', 'CV'] == float('inf')  # CV should be inf when mean is 0
    assert stats.loc['Communication_density', 'CV'] == float('inf')  # CV should be inf when mean is 0

def test_calculate_newness_proportion_input_types(analyzer):
    """Test _calculate_newness_proportion with different input types"""
    # Test with list of numpy arrays
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    result_list = analyzer._calculate_newness_proportion(vectors, 2)
    assert isinstance(result_list, float)
    assert 0 <= result_list <= 1

    # Test with list of pandas Series
    series_list = [pd.Series([1.0, 0.0]), pd.Series([0.0, 1.0]), pd.Series([1.0, 1.0])]
    result_series = analyzer._calculate_newness_proportion(series_list, 2)
    assert isinstance(result_series, float)
    assert 0 <= result_series <= 1

    # Test with numpy array
    array_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result_array = analyzer._calculate_newness_proportion(array_vectors, 2)
    assert isinstance(result_array, float)
    assert 0 <= result_array <= 1

    # Test with mixed types
    mixed_vectors = [
        [1.0, 0.0],  # Regular list
        pd.Series([0.0, 1.0]),  # pandas Series
        np.array([1.0, 1.0])  # numpy array
    ]
    result_mixed = analyzer._calculate_newness_proportion(mixed_vectors, 2)
    assert isinstance(result_mixed, float)
    assert 0 <= result_mixed <= 1

    # Test with regular Python lists
    list_vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # All regular Python lists
    result_list_only = analyzer._calculate_newness_proportion(list_vectors, 2)
    assert isinstance(result_list_only, float)
    assert 0 <= result_list_only <= 1
