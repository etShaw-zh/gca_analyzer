"""
GCA (Group Conversation Analysis) Analyzer Module

This module provides functionality for analyzing group conversations,
including participant interactions, metrics calculation, and visualization.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: MIT
"""

import pandas as pd
from typing import Tuple, List, Dict, Any
from .text_processor import TextProcessor
from .logger import logger
import numpy as np

class GCAAnalyzer:
    """
    Main analyzer class for group conversation analysis.
    
    This class integrates text processing, metrics calculation, and visualization
    components to provide comprehensive analysis of group conversations.
    """

    def __init__(self):
        """Initialize the GCA Analyzer with required components."""
        logger.info("Initializing GCA Analyzer")
        self.text_processor = TextProcessor()
        logger.debug("Components initialized successfully")

    def participant_pre(self, video_id: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[int], int, int, pd.DataFrame]:
        """
        Preprocess participant data.
        
        Args:
            video_id: ID of the video to analyze
            data: DataFrame containing all participant data
            
        Returns:
            Tuple containing:
            - Preprocessed DataFrame
            - List of participant IDs
            - List of contribution sequence numbers
            - Number of participants
            - Number of contributions
            - Participation matrix
        """
        # Filter data for current video
        current_data = data[data.video_id == video_id].copy()
        
        if current_data.empty:
            raise ValueError(f"No data found for video_id: {video_id}")
            
        # Validate required columns
        required_columns = ['person_id', 'time', 'text']
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Sort by time and add contribution sequence number
        current_data = current_data.sort_values('time').reset_index(drop=True)
        current_data['seq_num'] = range(1, len(current_data) + 1)
        
        # Clean text data
        current_data['text_clean'] = current_data.text.apply(self.text_processor.chinese_word_cut)
        
        # Get unique participants and sequence numbers
        person_list = current_data.person_id.unique().tolist()
        seq_list = current_data.seq_num.tolist()
        
        # Calculate dimensions
        k = len(person_list)  # Number of participants
        n = len(seq_list)    # Number of contributions
        
        # Create participation matrix M using sequence numbers
        M = pd.DataFrame(0, index=person_list, columns=seq_list)
        for _, row in current_data.iterrows():
            M.loc[row.person_id, row.seq_num] = 1
        
        return current_data, person_list, seq_list, k, n, M

    def get_best_window_num(self, seq_list: List[int], M: pd.DataFrame, best_window_indices: float = 0.3, min_num: int = 2, max_num: int = 10) -> int:
        """
        Find the optimal window size for analysis.
        
        Args:
            seq_list: List of contribution sequence numbers
            M: Participation matrix
            best_window_indices: Target participation threshold
            min_num: Minimum window size
            max_num: Maximum window size
            
        Returns:
            int: Optimal window size
        """
        # Validate input parameters
        if min_num > max_num:
            raise ValueError("min_num cannot be greater than max_num")
            
        if not (0 <= best_window_indices <= 1):
            raise ValueError("best_window_indices must be between 0 and 1")
            
        # Handle extreme thresholds
        if best_window_indices == 0:
            return min_num
        if best_window_indices == 1:
            return max_num
            
        n = len(seq_list)
        for w in range(min_num, max_num):
            found_valid_window = False
            for t in range(n):
                window_end = t + w
                if window_end > n:
                    break
                    
                # Get the sequence numbers for the window
                window_seqs = seq_list[t:window_end]
                current_user_data = pd.DataFrame()
                current_user_data['temp'] = M[window_seqs].apply(lambda x: x.sum(), axis=1)
                percen = len(current_user_data[current_user_data['temp'] >= 2]) / len(current_user_data['temp'])
                
                if percen >= best_window_indices:
                    found_valid_window = True
                    break
                    
            if found_valid_window:
                return w
        return max_num

    def get_Ksi_lag(self, best_window_length: int, person_list: List[str], k: int,
                    seq_list: List[int], M: pd.DataFrame, cosine_similarity_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Ksi lag matrix for interaction analysis.
        
        Args:
            best_window_length: Optimal window size
            person_list: List of participants
            k: Number of participants
            seq_list: List of contribution sequence numbers
            M: Participation matrix
            cosine_similarity_matrix: Matrix of cosine similarities
            
        Returns:
            pd.DataFrame: Ksi lag matrix
        """
        Ksi_lag = pd.DataFrame(0, index=person_list, columns=person_list)
        w = best_window_length
        
        for tao in range(1, w + 1):
            _Ksi_lag = pd.DataFrame(0, index=person_list, columns=person_list)
            for i in range(k):
                a = person_list[i]
                for j in range(k):
                    b = person_list[j]
                    Pab_tao = 0
                    Sab_sum = 0
                    
                    # Only consider contributions with sequence numbers greater than or equal to tau
                    valid_seqs = [seq for seq in seq_list if seq >= tao]
                    for seq in valid_seqs:
                        # Ensure that seq-tao exists in the sequence list
                        if (seq-tao) in seq_list:
                            Pab_tao += M.loc[a, seq-tao] * M.loc[b, seq]
                            if seq != seq_list[-1]:
                                continue
                            
                            if Pab_tao > 0:
                                # Calculate the sum of cosine similarities
                                for s in valid_seqs:
                                    if (s-tao) in seq_list:
                                        Sab_sum += M.loc[a, s-tao] * M.loc[b, s] * \
                                                cosine_similarity_matrix.loc[s-tao, s]
                    
                    if Pab_tao > 0:
                        _Ksi_lag.loc[a, b] = Sab_sum / Pab_tao
                                
            Ksi_lag += _Ksi_lag
            
        return Ksi_lag * (1/w)

    def cosine_similarity(self, vec1: List[Tuple[int, float]], 
                         vec2: List[Tuple[int, float]]) -> float:
        """
        Calculate cosine similarity between two sparse vectors.
        
        Args:
            vec1: First sparse vector as list of (index, value) tuples
            vec2: Second sparse vector as list of (index, value) tuples
            
        Returns:
            float: Cosine similarity between the vectors
        """
        # Convert sparse vectors to dictionaries for easier lookup
        dict1 = dict(vec1)
        dict2 = dict(vec2)
        
        # Get all indices
        indices = set(dict1.keys()) | set(dict2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = sum(dict1.get(i, 0) * dict2.get(i, 0) for i in indices)
        mag1 = np.sqrt(sum(v * v for v in dict1.values()))
        mag2 = np.sqrt(sum(v * v for v in dict2.values()))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
            
        return dot_product / (mag1 * mag2)

    def analyze_video(self, video_id: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze video conversation data based on the paper's formulas.
        
        Args:
            video_id: Video ID
            data: DataFrame containing conversation data
            
        Returns:
            pd.DataFrame: Analysis results for each participant
        """
        # Preprocess data
        current_data, person_list, seq_list, k, n, M = self.participant_pre(video_id, data)
        
        # Initialize result DataFrame
        student = pd.DataFrame(0.0, 
                             index=person_list,
                             columns=['Pa', 'Pa_average', 'Pa_std', 'video_id', 'Pa_hat',
                                    'Internal_cohesion', 'Overall_responsivity',
                                    'Social_impact', 'Newness', 'Communication_density'],
                             dtype=float)
        
        student['video_id'] = video_id
        
        # Calculate participation metrics (based on formulas 4 and 5)
        for person in person_list:
            # ||Pa|| = Σ pa(t) (formula 4)
            student.loc[person, 'Pa'] = M.loc[person].sum()
            # p̄a = (1/n)||Pa|| (formula 5)
            student.loc[person, 'Pa_average'] = student.loc[person, 'Pa'] / n
            
        # Calculate participation standard deviation (formula 6)
        for person in person_list:
            variance = 0
            for seq in seq_list:
                variance += (M.loc[person, seq] - student.loc[person, 'Pa_average'])**2
            student.loc[person, 'Pa_std'] = np.sqrt(variance / (n-1))
            
        # Calculate relative participation (modified formula 9)
        student['Pa_hat'] = (student['Pa_average'] - 1/k) / (1/k)
        
        # Process text and calculate vectors
        vector, dataset = self.text_processor.doc2vector(current_data.text_clean)
        
        # Calculate cosine similarity matrix
        cosine_similarity_matrix = pd.DataFrame(np.zeros((len(seq_list), len(seq_list)), dtype=float), 
                                              index=seq_list, columns=seq_list)
        for i in range(len(vector)):
            for j in range(len(vector)):
                cosine_similarity_matrix.iloc[i, j] = self.cosine_similarity(vector[i], vector[j])
        
        # Get optimal window size
        w = self.get_best_window_num(seq_list, M)
        
        # Calculate Cross-cohesion (formulas 15 and 16)
        cross_cohesion = pd.DataFrame(0.0, index=person_list, columns=person_list)
        for a in person_list:
            for b in person_list:
                for tau in range(1, w+1):
                    Pab_tau = 0
                    Sab_sum = 0
                    
                    # 只考虑序号大于等于tau的贡献
                    valid_seqs = [seq for seq in seq_list if seq >= tau]
                    for seq in valid_seqs:
                        # 确保seq-tau存在于序号列表中
                        if (seq-tau) in seq_list:
                            Pab_tau += M.loc[a, seq-tau] * M.loc[b, seq]
                            if seq != seq_list[-1]:
                                continue
                            
                            if Pab_tau > 0:
                                # 计算相似度总和
                                for s in valid_seqs:
                                    if (s-tau) in seq_list:
                                        Sab_sum += M.loc[a, s-tau] * M.loc[b, s] * \
                                                cosine_similarity_matrix.loc[s-tau, s]
                    
                    if Pab_tau > 0:
                        cross_cohesion.loc[a, b] += Sab_sum / Pab_tau
        cross_cohesion = cross_cohesion / w
        
        # Calculate Overall responsivity (formula 19)
        for person in person_list:
            responsivity_sum = 0
            for other in person_list:
                if other != person:
                    responsivity_sum += cross_cohesion.loc[person, other]
            student.loc[person, 'Overall_responsivity'] = responsivity_sum / (k-1)
        
        # Calculate Internal cohesion (formula 18)
        for person in person_list:
            student.loc[person, 'Internal_cohesion'] = cross_cohesion.loc[person, person]
        
        # Calculate Social impact (formula 20)
        for person in person_list:
            impact_sum = 0
            for other in person_list:
                if other != person:
                    impact_sum += cross_cohesion.loc[other, person]
            student.loc[person, 'Social_impact'] = impact_sum / (k-1)
        
        # Calculate Newness (formulas 25 and 26)
        for person in person_list:
            person_messages = current_data[current_data.person_id == person]
            if not person_messages.empty:
                newness_sum = 0
                for idx, row in person_messages.iterrows():
                    # Get previous message vectors
                    historical_vectors = [vector[i] for i in range(idx)]
                    if historical_vectors:
                        try:
                            # Calculate orthogonal projection
                            current_vector = np.array([v[1] for v in vector[idx]])
                            # Ensure all historical vectors have the same length as current vector
                            max_len = len(current_vector)
                            historical_matrix = []
                            for vec in historical_vectors:
                                vec_values = [v[1] for v in vec]
                                # Pad with zeros if necessary
                                if len(vec_values) < max_len:
                                    vec_values.extend([0] * (max_len - len(vec_values)))
                                elif len(vec_values) > max_len:
                                    vec_values = vec_values[:max_len]
                                historical_matrix.append(vec_values)
                            
                            historical_matrix = np.array(historical_matrix)
                            if historical_matrix.size > 0:  # Check if we have any valid vectors
                                # Use QR decomposition to calculate orthogonal complement space projection
                                Q, R = np.linalg.qr(historical_matrix.T)
                                proj = current_vector - Q @ (Q.T @ current_vector)
                                # n(ct) (formula 25)
                                newness = np.linalg.norm(proj) / (np.linalg.norm(proj) + np.linalg.norm(current_vector))
                                newness_sum += newness
                        except (ValueError, np.linalg.LinAlgError) as e:
                            # If we encounter any linear algebra errors, skip this vector
                            continue
                # Na (formula 26)
                student.loc[person, 'Newness'] = newness_sum / student.loc[person, 'Pa']
                
        # Calculate Communication density (formulas 27 and 28)
        for person in person_list:
            person_messages = current_data[current_data.person_id == person]
            if not person_messages.empty:
                density_sum = 0
                for idx, row in person_messages.iterrows():
                    # Di (formula 27)
                    vector_norm = np.linalg.norm([v[1] for v in vector[idx]])
                    word_length = len(dataset[idx])
                    if word_length > 0:
                        density_sum += vector_norm / word_length
                # D̄a (formula 28)
                student.loc[person, 'Communication_density'] = density_sum / student.loc[person, 'Pa']
                
        return student
