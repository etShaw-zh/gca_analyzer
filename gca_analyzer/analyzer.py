"""
GCA (Group Conversation Analysis) Analyzer Module

This module provides functionality for analyzing group conversations,
including participant interactions, metrics calculation, and visualization.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from .llm_processor import LLMTextProcessor
from .utils import cosine_similarity_matrix
from .logger import logger
from .config import Config, default_config

class GCAAnalyzer:
    """
    Main analyzer class for group conversation analysis.

    This class integrates text processing, metrics calculation, and visualization
    components to provide comprehensive analysis of group conversations.
    Supports multiple languages through advanced LLM-based text processing.

    Attributes:
        _config (Config): Configuration instance.
        llm_processor (LLMTextProcessor): LLM processor instance.
    """

    def __init__(
        self,
        llm_processor: LLMTextProcessor = None,
        config: Config = None
    ):
        """Initialize the GCA Analyzer with required components.

        Args:
            llm_processor (LLMProcessor, optional): LLM processor instance.
                Defaults to None.
            config (Config, optional): Configuration instance.
                Defaults to None.
        """
        self._config = config or default_config
        self.llm_processor = llm_processor or LLMTextProcessor(
            model_name=self._config.model.model_name,
            mirror_url=self._config.model.mirror_url
        )
        logger.info("Initializing GCA Analyzer")
        logger.info("Using LLM-based text processor for multi-language support")
        logger.debug("Components initialized successfully")

    def participant_pre(
        self,
        conversation_id: str,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], List[int], int, int, pd.DataFrame]:
        """Preprocess participant data.

        Args:
            conversation_id: Unique identifier for the conversation
            data: DataFrame containing all participant data

        Returns:
            Tuple containing:
            - Preprocessed DataFrame
            - List of participant IDs
            - List of contribution sequence numbers
            - Number of participants
            - Number of contributions
            - Participation matrix

        Raises:
            ValueError: If no data found for conversation_id or missing required columns
        """
        # Filter data for current conversation
        current_data = data[data.conversation_id == conversation_id].copy()
        
        if current_data.empty:
            raise ValueError(f"No data found for conversation_id: {conversation_id}")
            
        # Validate required columns
        required_columns = ['conversation_id', 'person_id', 'time', 'text']
        missing_columns = [
            col for col in required_columns if col not in current_data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Sort by time and add contribution sequence number
        current_data = current_data.sort_values('time').reset_index(drop=True)
        current_data['seq_num'] = range(1, len(current_data) + 1)
        
        # Get unique participants and sequence numbers
        person_list = sorted(current_data.person_id.unique())
        seq_list = sorted(current_data.seq_num.unique())
        
        # Calculate dimensions
        k = len(person_list)  # Number of participants
        n = len(seq_list)     # Number of contributions
        
        # Create participation matrix M using sequence numbers
        M = pd.DataFrame(0, index=person_list, columns=seq_list)
        for _, row in current_data.iterrows():
            M.loc[row.person_id, row.seq_num] = 1
        
        return current_data, person_list, seq_list, k, n, M

    def find_best_window_size(
        self,
        data: pd.DataFrame,
        best_window_indices: float = None,
        min_num: int = None,
        max_num: int = None
    ) -> int:
        """Find the optimal window size for analysis.

        Args:
            data: Input data for analysis
            best_window_indices: Target participation threshold
            min_num: Minimum window size
            max_num: Maximum window size

        Returns:
            Optimal window size

        Raises:
            ValueError: If min_num > max_num or best_window_indices not in [0,1]
        """
        best_window_indices = (
            best_window_indices or self._config.window.best_window_indices
        )
        min_num = min_num or self._config.window.min_window_size
        max_num = max_num or self._config.window.max_window_size
        
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
            
        n = len(data)
        # Group data by person_id to count contributions
        person_contributions = data.groupby('person_id')
        
        for w in range(min_num, max_num + 1):
            found_valid_window = False
            for t in range(n - w + 1):  # Adjust range to ensure valid windows
                window_data = data.iloc[t:t+w]
                
                # Count contributions per person in the window
                window_counts = window_data.groupby('person_id').size()
                # Calculate percentage of participants with >= 2 contributions
                active_participants = (window_counts >= 2).sum()
                total_participants = len(person_contributions)
                participation_rate = active_participants / total_participants
                
                if participation_rate >= best_window_indices:
                    found_valid_window = True
                    logger.info(f"Found valid window size: {w}")
                    return w
            
            if not found_valid_window and w == max_num:
                logger.info(
                    f"No valid window size found between {min_num} and {max_num}"
                )
                return max_num
        
        return max_num

    def get_Ksi_lag(
        self,
        best_window_length: int,
        person_list: List[str],
        k: int,
        seq_list: List[int],
        M: pd.DataFrame,
        cosine_similarity_matrix: pd.DataFrame
    ) -> pd.DataFrame:
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
        # Formula 15: Cross-cohesion function
        def calculate_ksi_ab_tau(
            a: str,
            b: str,
            tau: int,
            M: pd.DataFrame,
            cosine_similarity_matrix: pd.DataFrame,
            seq_list: List[int]
        ) -> float:
            """Calculate cross-cohesion for participants a and b at lag tau."""
            Pab_tau = 0.0
            Sab_sum = 0.0
            
            # Convert seq_list to sorted list to ensure proper indexing
            sorted_seqs = sorted(seq_list)
            
            for i, t in enumerate(sorted_seqs):
                if i >= tau:  # Check if we have enough previous messages
                    prev_t = sorted_seqs[i - tau]  # Get the lagged sequence number
                    Pab_tau += float(M.loc[a, prev_t]) * float(M.loc[b, t])
            
            if Pab_tau == 0:
                return 0.0
                
            for i, t in enumerate(sorted_seqs):
                if i >= tau:
                    prev_t = sorted_seqs[i - tau]
                    Sab_sum += float(M.loc[a, prev_t]) * float(M.loc[b, t]) * \
                              float(cosine_similarity_matrix.loc[prev_t, t])
            
            return Sab_sum / Pab_tau

        # Initialize w-spanning cross-cohesion matrix with float dtype
        X_tau = pd.DataFrame(0.0, index=person_list, columns=person_list, dtype=float)
        w = best_window_length
        
        # Calculate cross-cohesion for each tau and accumulate
        for tau in range(1, w + 1):
            for a in person_list:
                for b in person_list:
                    result = calculate_ksi_ab_tau(
                        a, b, tau, M, cosine_similarity_matrix, seq_list
                    )
                    X_tau.loc[a, b] = X_tau.loc[a, b] + result
        
        # Formula 17: Responsivity across w
        R_w = X_tau.multiply(1.0/w)
        
        return R_w

    def calculate_cohesion_response(
        self,
        person_list: List[str],
        k: int,
        R_w: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate cohesion and response matrices for interaction analysis.

        Args:
            person_list: List of participants
            k: Number of participants
            R_w: Responsivity across w

        Returns:
            pd.DataFrame: Cohesion and response matrices
        """
        metrics_df = pd.DataFrame(
            index=person_list,
            columns=['Internal_cohesion', 'Overall_responsivity', 'Social_impact']
        )

        for person in person_list:
            # Calculate Internal cohesion with w-spanning (Formula 18)
            metrics_df.loc[person, 'Internal_cohesion'] = R_w.loc[person, person]
        
            # Calculate Overall responsivity with w-spanning (Formula 19)
            responsivity_sum = sum(
                R_w.loc[person, other] for other in person_list if other != person
            )
            metrics_df.loc[person, 'Overall_responsivity'] = responsivity_sum / (k-1)
        
            # Calculate Social impact with w-spanning (Formula 20)
            impact_sum = sum(
                R_w.loc[other, person] for other in person_list if other != person
            )
            metrics_df.loc[person, 'Social_impact'] = impact_sum / (k-1)

        return (
            metrics_df['Internal_cohesion'],
            metrics_df['Overall_responsivity'],
            metrics_df['Social_impact']
        )

    def _calculate_lsa_metrics(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        current_idx: int
    ) -> Tuple[float, float]:
        """
        Calculate LSA given-new metrics for a contribution.

        Args:
            vectors: List of document vectors
            texts: List of corresponding texts
            current_idx: Index of current contribution

        Returns:
            Tuple containing:
            - Proportion of new content
            - Communication density
        """
        if current_idx == 0:
            return 1.0, np.linalg.norm(vectors[0])
            
        # Formula 21: Get previous contribution vectors
        prev_vectors = np.array(vectors[:current_idx])
        
        # Formula 22: Project onto previous contributions space
        # Calculate projection matrix for the subspace
        U, _, _ = np.linalg.svd(prev_vectors.T, full_matrices=False)
        proj_matrix = U @ U.T
        
        # Current vector
        d_i = vectors[current_idx]
        
        # Formula 22: Get given component
        g_i = proj_matrix @ d_i
        
        # Formula 23: Get new component
        n_i = d_i - g_i
        
        # Formula 25: Calculate newness proportion
        n_norm = np.linalg.norm(n_i)
        g_norm = np.linalg.norm(g_i)
        if n_norm + g_norm > 0:
            n_c_t = n_norm / (n_norm + g_norm)
        else:
            n_c_t = 0.0
            
        # Formula 27: Normalize density by contribution length
        current_text_length = len(texts[current_idx])
        if current_text_length > 0:
            D_i = np.linalg.norm(d_i) / current_text_length
        else:
            D_i = 0.0
        
        return n_c_t, D_i

    def calculate_given_new_dict(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        current_data: pd.DataFrame
    ) -> Tuple[dict, dict]:
        n_c_t_dict = {}  # Store newness values for later averaging
        D_i_dict = {}  # Store density values for later averaging
        
        for idx in range(len(vectors)):
            try:
                n_c_t, D_i = self._calculate_lsa_metrics(vectors, texts, idx)
                current_person = current_data.iloc[idx].person_id
                    
                if current_person not in n_c_t_dict:
                    n_c_t_dict[current_person] = []
                if current_person not in D_i_dict:
                    D_i_dict[current_person] = []
                        
                n_c_t_dict[current_person].append(n_c_t)
                D_i_dict[current_person].append(D_i)
                    
            except Exception as e:
                logger.error(
                    f"Error calculating LSA metrics for contribution {idx}: {str(e)}"
                )
                continue
        
        return n_c_t_dict, D_i_dict

    def calculate_given_new_averages(
        self,
        person_list: List[str],
        n_c_t_dict: dict,
        D_i_dict: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate average LSA metrics (newness and communication density) per person.

        Args:
            person_list: List of participant IDs
            newness_dict: Dictionary of newness values per person
            density_dict: Dictionary of density values per person

        Returns:
            DataFrame containing averaged LSA metrics
        """
        metrics_df = pd.DataFrame(
            0.0,
            index=person_list,
            columns=['Newness', 'Communication_density'],
            dtype=float
        )

        for person in person_list:
            if person in n_c_t_dict:
                # Formula 26
                metrics_df.loc[person, 'Newness'] = np.mean(
                    n_c_t_dict.get(person, [0.0])
                )
                # Formula 28
                metrics_df.loc[person, 'Communication_density'] = np.mean(
                    D_i_dict.get(person, [0.0])
                )
            else:
                metrics_df.loc[person, 'Newness'] = 0.0
                metrics_df.loc[person, 'Communication_density'] = 0.0
                
        return (
            metrics_df['Newness'],
            metrics_df['Communication_density']
        )

    def analyze_conversation(
        self,
        conversation_id: str,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze a conversation's dynamics using GCA metrics.

        The following metrics are calculated according to the formulas in the paper:
        1. Participation Rate (Pa): ||Pa|| = sum(M_a,t) for t=1 to n (Formula 4)
        2. Average Participation Rate (p̄a): p̄a = (1/n)||Pa|| (Formula 5)
        3. Participation Standard Deviation (σa): σa = sqrt((1/(n-1))sum((M_a,t - p̄a)^2)) (Formula 6)
        4. Normalized Participation Rate (P̂a): P̂a = (p̄a - 1/k)/(1/k) (Formula 9)
        5. Cross-Cohesion Matrix (Ξ): Ξ_ab = (1/w)sum(sum(M_a,t-τ * M_b,t * S_t-τ,t)/sum(M_a,t-τ * M_b,t)) (Formula 16)
        6. Internal Cohesion (Ca): Ca = Ξ_aa (Formula 18)
        7. Overall Responsivity (Ra): Ra = (1/(k-1))sum(Ξ_ab) for b≠a (Formula 19)
        8. Social Impact (Ia): Ia = (1/(k-1))sum(Ξ_ba) for b≠a (Formula 20)
        9. Message Newness (n(ct)): n(ct) = ||proj_⊥H_t(ct)|| / (||proj_⊥H_t(ct)|| + ||ct||) (Formula 25)
        10. Communication Density (Di): Di = ||ct|| / Lt (Formula 27)

        Args:
            conversation_id (str): Unique identifier for the conversation to be analyzed.
            data (pd.DataFrame): DataFrame containing conversation data with the following required columns:
                - person_id: Identifier for each participant
                - text: The content of each message
                - timestamp: Timestamp of each message
                - seq: Sequential number of the message in conversation

        Returns:
            pd.DataFrame: A DataFrame containing calculated GCA metrics for each participant with columns:
                - conversation_id: The input conversation identifier
                - Pa: Raw participation count (Formula 4)
                - Pa_average: Average participation rate (Formula 5)
                - Pa_std: Standard deviation of participation (Formula 6)
                - Pa_hat: Normalized participation rate (Formula 9)
                - Internal_cohesion: Self-response coherence measure (Formula 18)
                - Overall_responsivity: Response behavior to others (Formula 19)
                - Social_impact: Impact on others' responses (Formula 20)
                - Newness: Average message novelty (Formula 25)
                - Communication_density: Average message information density (Formula 27)

        Note:
            All metrics are calculated based on both message frequency and content analysis
            using language model embeddings for semantic understanding.
        """
        # Preprocess data and get basic information
        current_data, person_list, seq_list, k, n, M = self.participant_pre(
            conversation_id, data
        )
        
        # Initialize result DataFrame
        metrics_df = pd.DataFrame(
            0.0,
            index=person_list,
            columns=[
                'conversation_id', 'Pa', 'Pa_average', 'Pa_std', 'Pa_hat',
                'Internal_cohesion', 'Overall_responsivity',
                'Social_impact', 'Newness', 'Communication_density'
            ],
            dtype=float
        )
        metrics_df['conversation_id'] = conversation_id
        
        # Calculate participation metrics (Formula 4 and 5)
        for person in person_list:
            # Pa = sum(M_a) (Formula 4)
            metrics_df.loc[person, 'Pa'] = M.loc[person].sum()
            # p̄a = (1/n)||Pa|| (Formula 5)
            metrics_df.loc[person, 'Pa_average'] = metrics_df.loc[person, 'Pa'] / n
            
        # Calculate participation standard deviation (Formula 6)
        for person in person_list:
            variance = 0
            for seq in seq_list:
                variance += (
                    M.loc[person, seq] - metrics_df.loc[person, 'Pa_average']
                )**2
            metrics_df.loc[person, 'Pa_std'] = np.sqrt(variance / (n-1))
            
        # Calculate relative participation (Formula 9)
        metrics_df['Pa_hat'] = (
            metrics_df['Pa_average'] - 1/k
        ) / (1/k)
        
        # Process text and calculate vectors
        texts = current_data.text.to_list()
        vectors = self.llm_processor.doc2vector(texts)
        
        # Calculate cross-cohesion and w-spanning responsivity matrix
        w = self.find_best_window_size(current_data)
        logger.info(f"Using window size: {w}")
        
        # Create cosine similarity matrix for sequential contributions
        cosine_matrix = cosine_similarity_matrix(
            vectors, seq_list, current_data
        )
        
        R_w = self.get_Ksi_lag(
            w, person_list, k, seq_list, M, cosine_matrix
        )
        
        # Calculate Internal cohesion (Formula 18), Overall responsivity (Formula 19), Social impact (Formula 20) with w-spanning
        metrics_df['Internal_cohesion'], metrics_df['Overall_responsivity'], metrics_df['Social_impact'] = \
            self.calculate_cohesion_response(
                person_list=person_list, k=k, R_w=R_w
            )

        # Calculate newness and communication density (Formula 25 and 27) without w-spanning
        n_c_t_dict, D_i_dict = self.calculate_given_new_dict(
            vectors=vectors,
            texts=texts,
            current_data=current_data
        )

        # Calculate average metrics per person (Formula 26 and 28)
        metrics_df['Newness'], \
        metrics_df['Communication_density'] = self.calculate_given_new_averages(
            person_list=person_list,
            n_c_t_dict=n_c_t_dict,
            D_i_dict=D_i_dict
        )
        return metrics_df