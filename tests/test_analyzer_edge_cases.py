import pytest
import pandas as pd
import numpy as np
from gca_analyzer.analyzer import GCAAnalyzer
from gca_analyzer.llm_processor import LLMTextProcessor
import tempfile

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
    # Create empty lists for B to test edge case
    n_c_t_dict = {'A': [0.5], 'B': []}
    D_i_dict = {'A': [0.3], 'B': []}
    weighted_n_c_t_dict = {'A': [0.5], 'B': []}
    weighted_D_i_dict = {'A': [0.3], 'B': []}
    
    # Initialize metrics DataFrame
    metrics_df = pd.DataFrame(index=person_list)
    
    newness, density, w_newness, w_density = analyzer.calculate_personal_given_new_averages(
        person_list, n_c_t_dict, D_i_dict, weighted_n_c_t_dict, weighted_D_i_dict
    )
    
    # Test non-empty case
    assert newness['A'] == pytest.approx(0.5)
    assert density['A'] == pytest.approx(0.3)
    assert w_newness['A'] == pytest.approx(0.5)
    assert w_density['A'] == pytest.approx(0.3)
    
    # Test empty case - should be 0.0 for empty lists
    assert pd.isna(newness['B']) or newness['B'] == pytest.approx(0.0)
    assert pd.isna(density['B']) or density['B'] == pytest.approx(0.0)
    assert pd.isna(w_newness['B']) or w_newness['B'] == pytest.approx(0.0)
    assert pd.isna(w_density['B']) or w_density['B'] == pytest.approx(0.0)

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
    # Create a test conversation with edge cases
    conversation_id = "test_conv"
    all_metrics = {
        conversation_id: pd.DataFrame({
            'participation': [0.0, 1.0, None],
            'responsivity': [0.5, None, 0.7],
            'internal_cohesion': [None, 0.3, 0.8],
            'social_impact': [0.2, 0.6, None],
            'newness': [0.1, None, 0.9],
            'comm_density': [None, 0.4, 0.5]
        })
    }
    
    # Test with temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        stats_df = analyzer.calculate_descriptive_statistics(conversation_id, all_metrics, temp_dir)
    
    # Verify basic statistics are calculated correctly
    assert not stats_df.empty
    assert 'Mean' in stats_df.columns
    assert 'SD' in stats_df.columns
    assert 'Missing' in stats_df.columns

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

def test_calculate_newness_proportion_with_pandas_series():
    """Test calculation of newness proportion with pandas Series vectors."""
    analyzer = GCAAnalyzer()

    # Create test data with numpy arrays
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0])
    ]

    # Test with first vector (should return 1.0 as it's completely new)
    result = analyzer._calculate_newness_proportion(vectors, 0)
    assert np.isclose(result, 1.0)

    # Test with second vector (should be partially new)
    result = analyzer._calculate_newness_proportion(vectors, 1)
    assert result > 0 and result < 1

    # Test with third vector (should be partially new)
    result = analyzer._calculate_newness_proportion(vectors, 2)
    assert result > 0 and result < 1

    # Test with DataFrame input
    df_vectors = pd.DataFrame({
        'vectors': [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
    })
    
    # Test with first vector in DataFrame
    result = analyzer._calculate_newness_proportion(df_vectors['vectors'].values, 0)
    assert np.isclose(result, 1.0)

    # Test with second vector in DataFrame
    result = analyzer._calculate_newness_proportion(df_vectors['vectors'].values, 1)
    assert result > 0 and result < 1

    # Test with third vector in DataFrame
    result = analyzer._calculate_newness_proportion(df_vectors['vectors'].values, 2)
    assert result > 0 and result < 1

def test_calculate_text_given_new_df_with_empty_vectors():
    """Test calculate_text_given_new_df with empty vectors."""
    analyzer = GCAAnalyzer()
    # Create empty vectors and texts
    vectors = []
    texts = []
    current_data = pd.DataFrame()
    
    result = analyzer.calculate_text_given_new_df(vectors, texts, current_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_calculate_text_given_new_df_with_single_vector():
    """Test calculate_text_given_new_df with a single vector."""
    analyzer = GCAAnalyzer()
    # Create a single vector and text
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vectors = [vector]
    texts = ['test text']
    current_data = pd.DataFrame()
    
    result = analyzer.calculate_text_given_new_df(vectors, texts, current_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

def test_calculate_personal_given_new_with_invalid_input():
    """Test calculate_personal_given_new with invalid input."""
    analyzer = GCAAnalyzer()
    vectors = None
    texts = pd.Series(['test text'])
    current_data = pd.DataFrame()
    
    with pytest.raises(AttributeError):
        analyzer.calculate_personal_given_new(vectors, texts, current_data)
