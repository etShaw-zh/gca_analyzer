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
import os

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
            
        current_data['parsed_time'] = pd.to_datetime(current_data['time'], format='mixed')
        current_data = current_data.sort_values('parsed_time').reset_index(drop=True)
        current_data['seq_num'] = range(1, len(current_data) + 1)
        
        person_list = sorted(current_data.person_id.unique())
        seq_list = sorted(current_data.seq_num.unique())
        
        k = len(person_list)  
        n = len(seq_list)     
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
        person_contributions = data.groupby('person_id')
        
        for w in range(min_num, max_num + 1):
            found_valid_window = False
            for t in range(n - w + 1):
                window_data = data.iloc[t:t+w]
                
                window_counts = window_data.groupby('person_id').size()
                active_participants = (window_counts >= 2).sum() # at least 2 contributions in the window TODO: use a threshold
                total_participants = len(person_contributions)
                participation_rate = active_participants / total_participants
                
                if participation_rate >= best_window_indices:
                    found_valid_window = True
                    print(f"=== Found valid window size: {w} (current window threshold: {best_window_indices}) ===")
                    return w
            
            if not found_valid_window and w == max_num:
                print(f"=== No valid window size found between {min_num} and {max_num}, using max_num: {max_num} (current window threshold: {best_window_indices}) ===")
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
        Calculate LSA (Latent Semantic Analysis) given-new metrics for a contribution.

        This method computes two key metrics:
        1. Proportion of new content (n_c_t): Measures how much new information
           the current contribution brings compared to previous contributions.
        2. Communication density (D_i): Measures the information density of
           the current contribution, normalized by its length.

        The calculation involves projecting the current vector onto the subspace
        spanned by previous contribution vectors, then comparing the magnitudes
        of the projected (given) and residual (new) components.

        Args:
            vectors: List of document vectors, each representing a contribution.
            texts: List of corresponding text content for each contribution.
            current_idx: Index of the current contribution being analyzed.

        Returns:
            Tuple containing:
            - float: Proportion of new content (n_c_t), range [0, 1]
            - float: Communication density (D_i)

        Note:
            For the first contribution (current_idx == 0), n_c_t is set to 1.0
            (all content is new) and D_i is simply the norm of the first vector.
        """
        if current_idx == 0:
            return 1.0, np.linalg.norm(vectors[0])

        n_c_t = self._calculate_newness_proportion(vectors, current_idx)
        D_i = self._calculate_communication_density(vectors[current_idx], texts[current_idx])

        return n_c_t, D_i

    def _calculate_newness_proportion(self, vectors: List[np.ndarray], current_idx: int) -> float:
        """
        Calculate the proportion of new content in the current contribution.

        Args:
            vectors: List of document vectors.
            current_idx: Index of the current contribution.

        Returns:
            float: Proportion of new content (n_c_t), range [0, 1]
        """
        prev_vectors = np.array(vectors[:current_idx])
        d_i = vectors[current_idx]

        # Calculate projection matrix for the subspace
        U, _, _ = np.linalg.svd(prev_vectors.T, full_matrices=False)
        proj_matrix = U @ U.T

        # Get given and new components
        g_i = proj_matrix @ d_i
        n_i = d_i - g_i

        # Calculate newness proportion
        n_norm = np.linalg.norm(n_i)
        g_norm = np.linalg.norm(g_i)
        return n_norm / (n_norm + g_norm) if (n_norm + g_norm) > 0 else 0.0

    def _calculate_communication_density(self, vector: np.ndarray, text: str) -> float:
        """
        Calculate the communication density of a contribution.

        Args:
            vector: Document vector of the contribution.
            text: Corresponding text content of the contribution.

        Returns:
            float: Communication density (D_i)
        """
        text_length = len(text)
        return np.linalg.norm(vector) / text_length if text_length > 0 else 0.0

    def calculate_given_new_dict(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        current_data: pd.DataFrame
    ) -> Tuple[dict, dict]:
        n_c_t_dict = {}
        D_i_dict = {}

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
        current_data, person_list, seq_list, k, n, M = self.participant_pre(
            conversation_id, data
        )
        
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
        
        texts = current_data.text.to_list()
        vectors = self.llm_processor.doc2vector(texts)
        
        w = self.find_best_window_size(current_data)
        logger.info(f"Using window size: {w}")
        
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
        
        metrics_df = metrics_df.rename(columns={
            'Pa_hat': 'participation',
            'Overall_responsivity': 'responsivity',
            'Internal_cohesion': 'internal_cohesion',
            'Social_impact': 'social_impact',
            'Newness': 'newness',
            'Communication_density': 'comm_density'
        })
        
        return metrics_df

    def calculate_descriptive_statistics(
        self,
        all_metrics: dict,
        output_dir: str = None
    ) -> pd.DataFrame:
        """Calculate descriptive statistics for GCA measures.

        Args:
            all_metrics (dict): Dictionary of DataFrames containing metrics for each conversation.
            output_dir (str, optional): Directory to save the statistics CSV file.
                If None, the file will not be saved.

        Returns:
            pd.DataFrame: DataFrame containing descriptive statistics for each measure.
        """
        all_data = pd.concat(all_metrics.values())
        
        stats = pd.DataFrame({
            'Minimum': all_data.min(),
            'Median': all_data.median(),
            'M': all_data.mean(),
            'SD': all_data.std(),
            'Maximum': all_data.max(),
            'Count': all_data.count(),
            'Missing': all_data.isnull().sum(),
            'CV': all_data.std() / all_data.mean()
        }).round(2)
        
        print("=== Descriptive statistics for GCA measures ===")
        print("-" * 80)
        print("Measure".ljust(20), end='')
        print("Minimum  Median  M      SD     Maximum  Count  Missing  CV")
        print("-" * 80)
        for measure in stats.index:
            row = stats.loc[measure]
            cv_value = f"{row['CV']:.2f}" if row['CV'] < 10 else 'inf'
            print(f"{measure.replace('_', ' ').title().ljust(20)}"
                f"{row['Minimum']:7.2f}  "
                f"{row['Median']:6.2f}  "
                f"{row['M']:5.2f}  "
                f"{row['SD']:5.2f}  "
                f"{row['Maximum']:7.2f}  "
                f"{row['Count']:5.0f}  "
                f"{row['Missing']:7.0f}  "
                f"{cv_value:>5}"
            )
        print("-" * 80)
        
        if output_dir:
            output_file = os.path.join(output_dir, 'descriptive_statistics_gca.csv')
            stats.to_csv(output_file)
            print(f"Saved descriptive statistics to: {output_file}")
        
        return stats