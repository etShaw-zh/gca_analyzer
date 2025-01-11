import re
import jieba.posseg as psg
import jieba
from typing import List, Tuple, Dict, Set
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from .logger import logger

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with stop words and custom dictionaries."""
        logger.info("Initializing Text Processor")
        jieba.initialize()
        self.stop_words = self._load_stop_words()
        self.punctuation = set(string.punctuation + '。，？！：；""''【】「」『』（）〔〕［］《》〈〉{}｛｝⟨⟩')
        self.english_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with'
        }
        self.stop_words.update(self.english_stop_words)
        self.vectorizer = TfidfVectorizer(lowercase=True)
        logger.debug("Initialized stop words and vectorizer")

    def _load_stop_words(self) -> Set[str]:
        """Load stop words from file or return default set."""
        logger.debug("Loading stop words")
        default_stop_words = {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
            '之', '用', '于', '把', '等', '去', '又', '能', '好', '在',
            '还', '没', '要', '这', '那', '你', '我', '他', '她', '它'
        }
        try:
            with open('stop_words.txt', 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            logger.warning("Stop words file not found, using default stop words")
            return default_stop_words

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing unwanted characters and normalizing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        logger.info("Preprocessing text")
        if not isinstance(text, str):
            return ""
            
        # Convert to full-width characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        
        # Remove punctuation while preserving spaces
        text = ''.join(char if char not in self.punctuation else ' ' for char in text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        logger.debug("Preprocessed text")
        return text

    def chinese_word_cut(self, text: str) -> str:
        """
        Process Chinese text by segmenting it into words and removing stop words.
        For English text, simply split on spaces and remove stop words.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Processed text with words joined by spaces
        """
        logger.info("Processing Chinese text")
        # Preprocess text first
        text = self.preprocess_text(text)
        if not text:
            return ""
            
        # Check if text contains Chinese characters
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        word_list = []
        if has_chinese:
            # Process with jieba for Chinese text
            for word in jieba.cut(text):  
                word = word.strip().lower()
                
                # Skip empty strings and stop words
                if not word or word in self.stop_words:
                    continue
                    
                # Keep only Chinese characters, English letters, and numbers
                word = re.sub(r'[^\u4e00-\u9fff\w]', '', word)
                if word:
                    word_list.append(word)
        else:
            # Simple splitting for non-Chinese text
            for word in text.lower().split():
                word = word.strip()
                if word and word not in self.stop_words:
                    # Remove any remaining punctuation
                    word = re.sub(r'[^\w]', '', word)
                    if word:
                        word_list.append(word)
        
        logger.debug("Processed Chinese text")
        return ' '.join(word_list)

    def doc2vector(self, texts: List[str]) -> Tuple[List[List[Tuple[int, float]]], List[List[str]]]:
        """
        Convert a list of texts into TF-IDF vectors using scikit-learn.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            Tuple[List[List[Tuple[int, float]]], List[List[str]]]: 
                - TF-IDF vectors
                - Processed texts as lists of words
        """
        logger.info("Converting texts to vectors")
        # Handle empty or invalid texts
        texts = [text if isinstance(text, str) else "" for text in texts]
        
        # Split texts into words
        dataset = [text.split(' ') if text else [] for text in texts]
        
        # Create TF-IDF vectors
        if not any(texts):  # If all texts are empty
            return [], dataset
            
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Convert to the expected format (list of list of tuples)
            vectors = []
            for i in range(tfidf_matrix.shape[0]):
                row = tfidf_matrix[i].tocoo()
                vectors.append([(j, v) for j, v in zip(row.col, row.data)])
                
            logger.debug("Converted texts to vectors")
            return vectors, dataset
        except Exception as e:
            logger.error(f"Error converting texts to vectors: {str(e)}")
            raise
