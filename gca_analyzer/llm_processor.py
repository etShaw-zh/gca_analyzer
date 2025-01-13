"""
LLM Text Processor Module

This module provides advanced text processing capabilities using
large language models and transformers.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import subprocess
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .config import Config, default_config
from .logger import logger


class LLMTextProcessor:
    """Advanced text processing class using large language models.

    This class provides advanced text processing capabilities using transformer
    models for tasks such as embedding generation and similarity computation.

    Attributes:
        model_name (str): Name of the pretrained model being used
        mirror_url (str): URL of the model mirror, if any
        model (SentenceTransformer): The loaded transformer model
    """

    def __init__(
        self,
        model_name: str = None,
        mirror_url: str = None,
        config: Config = None
    ):
        """Initialize the LLM text processor.

        Args:
            model_name: Name of the pretrained model to use. Default is a
                multilingual model that supports 50+ languages including
                English, Chinese, Spanish, etc.
            mirror_url: Optional mirror URL for downloading models. If provided,
                will use this instead of the default Hugging Face server.
                For ModelScope models, use: "https://modelscope.cn/models"
            config: Configuration instance
        """
        self._config = config or default_config
        self.model_name = model_name or self._config.model.model_name
        self.mirror_url = mirror_url or self._config.model.mirror_url
        
        logger.info(
            f"Initializing LLM Text Processor with model: {self.model_name}"
        )
        try:
            if self.mirror_url and "modelscope.cn" in self.mirror_url:
                self._init_modelscope_model()
            else:
                self._init_huggingface_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _init_modelscope_model(self):
        """Initialize model from ModelScope."""
        try:
            from modelscope import snapshot_download
            # Download model to local cache
            model_dir = snapshot_download(self.model_name)
            self.model = SentenceTransformer(model_dir)
            logger.info(
                f"Successfully loaded model from ModelScope: {self.model_name}"
            )
        except ImportError:
            logger.warning("ModelScope not installed. Installing packages...")
            subprocess.check_call(["pip", "install", "modelscope"])
            from modelscope import snapshot_download
            model_dir = snapshot_download(self.model_name)
            self.model = SentenceTransformer(model_dir)
            logger.info(
                f"Successfully loaded model from ModelScope: {self.model_name}"
            )

    def _init_huggingface_model(self):
        """Initialize model from Hugging Face."""
        if self.mirror_url:
            import os
            os.environ['HF_ENDPOINT'] = self.mirror_url
            logger.info(f"Using custom mirror: {self.mirror_url}")
        
        self.model = SentenceTransformer(self.model_name)
        logger.info("Successfully loaded the model")

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts using the loaded model.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors as numpy arrays
        """
        try:
            # Generate vectors using the model
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            # Ensure each vector is a numpy array
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [
                np.zeros(self.model.get_sentence_embedding_dimension())
                for _ in texts
            ]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.get_embeddings([text1, text2])
            if len(embeddings) < 2:
                return 0.0
            
            # Convert to torch tensors
            emb1 = torch.tensor(embeddings[0])
            emb2 = torch.tensor(embeddings[1])
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0),
                emb2.unsqueeze(0)
            )
            return float(similarity[0])
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0

    def doc2vector(self, texts: List[str]) -> List[np.ndarray]:
        """Convert texts to vectors using transformer embeddings.

        Args:
            texts: List of input texts

        Returns:
            List of flattened embedding vectors as numpy arrays
        """
        logger.info("Converting texts to vectors using LLM")
        try:
            # Get embeddings and ensure they are flattened numpy arrays
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            vectors = [
                np.array(emb).flatten() if emb is not None
                else np.zeros(self.model.get_sentence_embedding_dimension())
                for emb in embeddings
            ]
            return vectors
        except Exception as e:
            logger.error(f"Error in doc2vector: {str(e)}")
            return [
                np.zeros(self.model.get_sentence_embedding_dimension())
                for _ in texts
            ]