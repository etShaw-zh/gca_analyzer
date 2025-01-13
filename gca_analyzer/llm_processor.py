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
from sentence_transformers import SentenceTransformer

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
        mirror_url: Optional[str] = None,
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
