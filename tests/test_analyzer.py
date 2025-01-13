"""
Test module for the GCA Analyzer

This module contains unit tests for the GCAAnalyzer class.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from gca_analyzer.analyzer import GCAAnalyzer
from gca_analyzer.llm_processor import LLMTextProcessor
from gca_analyzer.config import Config

@pytest.fixture
def sample_data():
    """Create sample conversation data for testing."""
    data = pd.DataFrame({
        'conversation_id': ['conv1'] * 6,
        'person_id': ['p1', 'p2', 'p1', 'p2', 'p3', 'p1'],
        'text': [
            'Hello everyone',
            'Hi there',
            'How are you?',
            'I am good',
            'Nice to meet you all',
            'Great to meet you too'
        ],
        'time': [
            '2025-01-13 10:00:00',
            '2025-01-13 10:01:00',
            '2025-01-13 10:02:00',
            '2025-01-13 10:03:00',
            '2025-01-13 10:04:00',
            '2025-01-13 10:05:00'
        ]
    })
    return data

@pytest.fixture
def analyzer():
    """Create a GCAAnalyzer instance for testing."""
    config = Config()
    llm_processor = LLMTextProcessor()
    return GCAAnalyzer(llm_processor=llm_processor, config=config)

def test_participant_pre(analyzer, sample_data):
    """Test participant preprocessing."""
    current_data, person_list, seq_list, k, n, M = analyzer.participant_pre(
        'conv1', sample_data
    )
    
    # Test output types
    assert isinstance(current_data, pd.DataFrame)
    assert isinstance(person_list, list)
    assert isinstance(seq_list, list)
    assert isinstance(k, int)
    assert isinstance(n, int)
    assert isinstance(M, pd.DataFrame)
    
    # Test dimensions and values
    assert len(person_list) == 3  # p1, p2, p3
    assert len(seq_list) == 6  # 6 messages
    assert k == 3  # 3 participants
    assert n == 6  # 6 messages
    assert M.shape == (3, 6)  # 3 participants x 6 messages
    
    # Test participation matrix values
    assert M.loc['p1', 1] == 1  # p1's first message
    assert M.loc['p2', 2] == 1  # p2's first message
    assert M.loc['p3', 5] == 1  # p3's message
    
def test_find_best_window_size(analyzer, sample_data):
    """Test finding the optimal window size."""
    # Test with default parameters
    w = analyzer.find_best_window_size(sample_data)
    assert isinstance(w, int)
    assert w >= 2  # Window size should be at least 2
    
    # Test with custom parameters
    w = analyzer.find_best_window_size(
        sample_data,
        best_window_indices=0.5,
        min_num=2,
        max_num=4
    )
    assert 2 <= w <= 4
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        analyzer.find_best_window_size(
            sample_data,
            min_num=4,
            max_num=2  # min_num > max_num
        )
    
    with pytest.raises(ValueError):
        analyzer.find_best_window_size(
            sample_data,
            best_window_indices=1.5  # > 1
        )

def test_get_Ksi_lag(analyzer, sample_data):
    """Test calculation of Ksi lag matrix."""
    # Prepare test data
    current_data, person_list, seq_list, k, n, M = analyzer.participant_pre(
        'conv1', sample_data
    )
    
    # Generate test vectors and similarity matrix
    texts = current_data.text.to_list()
    vectors = analyzer.llm_processor.doc2vector(texts)
    cosine_matrix = pd.DataFrame(
        np.eye(len(seq_list)),
        index=seq_list,
        columns=seq_list
    )
    
    # Test Ksi lag calculation
    R_w = analyzer.get_Ksi_lag(
        best_window_length=2,
        person_list=person_list,
        k=k,
        seq_list=seq_list,
        M=M,
        cosine_similarity_matrix=cosine_matrix
    )
    
    # Check output properties
    assert isinstance(R_w, pd.DataFrame)
    assert R_w.shape == (k, k)  # k x k matrix
    assert all(0 <= x <= 1 for x in R_w.values.flatten())  # values between 0 and 1

def test_calculate_cohesion_response(analyzer, sample_data):
    """Test calculation of cohesion and response metrics."""
    # Prepare test data
    current_data, person_list, seq_list, k, n, M = analyzer.participant_pre(
        'conv1', sample_data
    )
    
    # Create a test R_w matrix
    R_w = pd.DataFrame(
        np.random.random((k, k)),
        index=person_list,
        columns=person_list
    )
    
    # Calculate metrics
    internal_cohesion, overall_responsivity, social_impact = \
        analyzer.calculate_cohesion_response(person_list, k, R_w)
    
    # Test output types and ranges
    assert isinstance(internal_cohesion, pd.Series)
    assert isinstance(overall_responsivity, pd.Series)
    assert isinstance(social_impact, pd.Series)
    
    assert all(0 <= x <= 1 for x in internal_cohesion.values)
    assert all(0 <= x <= 1 for x in overall_responsivity.values)
    assert all(0 <= x <= 1 for x in social_impact.values)

def test_lsa_metrics(analyzer, sample_data):
    """Test LSA metrics calculations."""
    # Prepare test data
    texts = [
        "Hello world",
        "Hello there",
        "How are you"
    ]
    vectors = analyzer.llm_processor.doc2vector(texts)
    
    # Test newness calculation
    n_c_t = analyzer._calculate_newness_proportion(vectors, 1)
    assert isinstance(n_c_t, (float, np.float32))  # Allow both float types
    assert 0 <= n_c_t <= 1
    
    # Test communication density
    D_i = analyzer._calculate_communication_density(vectors[0], texts[0])
    assert isinstance(D_i, (float, np.float32))
    assert D_i >= 0

def test_analyze_conversation(analyzer, sample_data):
    """Test complete conversation analysis."""
    metrics_df = analyzer.analyze_conversation('conv1', sample_data)
    
    # Test output structure
    expected_columns = {
        'conversation_id', 'Pa', 'Pa_average', 'Pa_std',
        'participation', 'internal_cohesion', 'responsivity',
        'social_impact', 'newness', 'comm_density'
    }
    assert all(col in metrics_df.columns for col in expected_columns)
    
    # Test data integrity
    assert len(metrics_df) == 3  # 3 participants
    assert (metrics_df['Pa'] >= 0).all()  # non-negative participation
    assert ((metrics_df['newness'] >= 0) & (metrics_df['newness'] <= 1)).all()  # newness between 0 and 1
    assert (metrics_df['comm_density'] >= 0).all()  # non-negative density

def test_calculate_descriptive_statistics(analyzer, sample_data):
    """Test calculation of descriptive statistics."""
    # Analyze multiple conversations
    metrics1 = analyzer.analyze_conversation('conv1', sample_data)
    metrics2 = sample_data.copy()
    metrics2['conversation_id'] = 'conv2'
    metrics2 = analyzer.analyze_conversation('conv2', metrics2)
    
    all_metrics = {
        'conv1': metrics1,
        'conv2': metrics2
    }
    
    # Calculate statistics only for numeric columns
    numeric_columns = ['Pa', 'Pa_average', 'Pa_std', 'participation', 
                      'internal_cohesion', 'responsivity', 'social_impact', 
                      'newness', 'comm_density']
    
    # Merge data and keep only numeric columns
    all_data = pd.concat(all_metrics.values())[numeric_columns]
    
    stats_df = pd.DataFrame({
        'Minimum': all_data.min(),
        'Median': all_data.median(),
        'M': all_data.mean(),
        'SD': all_data.std(),
        'Maximum': all_data.max(),
        'Count': all_data.count(),
        'Missing': all_data.isnull().sum(),
        'CV': all_data.std() / all_data.mean()
    }).round(2)
    
    # Test output structure
    expected_columns = {
        'Minimum', 'Median', 'M', 'SD', 'Maximum',
        'Count', 'Missing', 'CV'
    }
    assert all(col in stats_df.columns for col in expected_columns)
    
    # Test data validity
    assert (stats_df['Minimum'] <= stats_df['Maximum']).all()
    assert (stats_df['Count'] > 0).all()
    assert (stats_df['SD'] >= 0).all()
