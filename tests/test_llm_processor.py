"""Test module for LLM text processor."""
import os
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np

from gca_analyzer.llm_processor import LLMTextProcessor
from gca_analyzer.config import Config


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    with patch('gca_analyzer.llm_processor.SentenceTransformer') as mock:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_modelscope():
    """Mock ModelScope for testing."""
    with patch('modelscope.snapshot_download') as mock:
        mock.return_value = '/tmp/model_dir'
        yield mock


def test_init_with_default_config(mock_sentence_transformer):
    """Test initialization with default configuration."""
    with patch('gca_analyzer.llm_processor.default_config') as mock_config:
        mock_config.model.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        mock_config.model.mirror_url = None
        processor = LLMTextProcessor()
        assert processor.model_name == 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        assert processor.mirror_url is None


def test_init_with_custom_config(mock_sentence_transformer):
    """Test initialization with custom configuration."""
    config = Config()
    config.model.model_name = 'custom-model'
    config.model.mirror_url = 'https://custom-mirror.com'
    processor = LLMTextProcessor(config=config)
    assert processor.model_name == 'custom-model'
    assert processor.mirror_url == 'https://custom-mirror.com'


def test_init_with_explicit_params(mock_sentence_transformer):
    """Test initialization with explicit parameters."""
    processor = LLMTextProcessor(
        model_name='explicit-model',
        mirror_url='https://explicit-mirror.com'
    )
    assert processor.model_name == 'explicit-model'
    assert processor.mirror_url == 'https://explicit-mirror.com'


def test_init_huggingface_model(mock_sentence_transformer):
    """Test initialization of Hugging Face model."""
    with patch('gca_analyzer.llm_processor.default_config') as mock_config:
        mock_config.model.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        mock_config.model.mirror_url = None
        processor = LLMTextProcessor()
        mock_sentence_transformer.assert_called_once_with(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )


def test_init_huggingface_model_with_mirror(mock_sentence_transformer):
    """Test initialization of Hugging Face model with mirror URL."""
    mirror_url = 'https://custom-mirror.com'
    with patch.dict(os.environ, clear=True):
        processor = LLMTextProcessor(mirror_url=mirror_url)
        assert os.environ.get('HF_ENDPOINT') == mirror_url


@patch('subprocess.check_call')
def test_init_modelscope_model(mock_subprocess, mock_modelscope, mock_sentence_transformer):
    """Test initialization of ModelScope model."""
    processor = LLMTextProcessor(
        model_name='damo/nlp_model',
        mirror_url='https://modelscope.cn/models'
    )
    mock_modelscope.assert_called_once_with('damo/nlp_model')
    mock_sentence_transformer.assert_called_once_with('/tmp/model_dir')


@patch('subprocess.check_call')
def test_init_modelscope_model_with_import_error(
    mock_subprocess, mock_modelscope, mock_sentence_transformer
):
    """Test ModelScope initialization with initial import error."""
    with patch('builtins.__import__') as mock_import:
        def mock_import_effect(*args, **kwargs):
            if args[0] == 'modelscope':
                if not mock_subprocess.called:  # First import before pip install
                    raise ImportError("No module named 'modelscope'")
                else:
                    # After pip install, return a mock module with snapshot_download
                    mock_module = MagicMock()
                    mock_module.snapshot_download = mock_modelscope
                    return mock_module
            return MagicMock()
        
        mock_import.side_effect = mock_import_effect
        
        # Mock the snapshot_download function to return a model directory
        mock_modelscope.return_value = '/tmp/model_dir'
        
        processor = LLMTextProcessor(
            model_name='damo/nlp_model',
            mirror_url='https://modelscope.cn/models'
        )
        
        # Verify pip install was called
        mock_subprocess.assert_called_once_with(
            ['pip', 'install', 'modelscope']
        )
        
        # Verify modelscope was used after installation
        mock_modelscope.assert_called_once_with('damo/nlp_model')
        mock_sentence_transformer.assert_called_once_with('/tmp/model_dir')


def test_doc2vector_success(mock_sentence_transformer):
    """Test successful text to vector conversion."""
    mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_sentence_transformer.return_value.encode.return_value = mock_embeddings
    mock_sentence_transformer.return_value.get_sentence_embedding_dimension.return_value = 2

    processor = LLMTextProcessor()
    texts = ['text1', 'text2']
    vectors = processor.doc2vector(texts)

    assert len(vectors) == 2
    assert all(isinstance(v, np.ndarray) for v in vectors)
    assert all(v.shape == (2,) for v in vectors)
    mock_sentence_transformer.return_value.encode.assert_called_once_with(
        texts, convert_to_numpy=True
    )


def test_doc2vector_with_error(mock_sentence_transformer):
    """Test text to vector conversion with error."""
    mock_sentence_transformer.return_value.encode.side_effect = Exception('Test error')
    mock_sentence_transformer.return_value.get_sentence_embedding_dimension.return_value = 2

    processor = LLMTextProcessor()
    texts = ['text1', 'text2']
    vectors = processor.doc2vector(texts)

    assert len(vectors) == 2
    assert all(isinstance(v, np.ndarray) for v in vectors)
    assert all(v.shape == (2,) for v in vectors)
    assert all(np.array_equal(v, np.zeros(2)) for v in vectors)


def test_doc2vector_with_none_embedding(mock_sentence_transformer):
    """Test text to vector conversion with None embedding."""
    mock_embeddings = [None, np.array([0.3, 0.4])]
    mock_sentence_transformer.return_value.encode.return_value = mock_embeddings
    mock_sentence_transformer.return_value.get_sentence_embedding_dimension.return_value = 2

    processor = LLMTextProcessor()
    texts = ['text1', 'text2']
    vectors = processor.doc2vector(texts)

    assert len(vectors) == 2
    assert all(isinstance(v, np.ndarray) for v in vectors)
    assert all(v.shape == (2,) for v in vectors)
    assert np.array_equal(vectors[0], np.zeros(2))  # First embedding was None
    assert np.array_equal(vectors[1], np.array([0.3, 0.4]))  # Second embedding was valid
