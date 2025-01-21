"""Group Conversation Analysis (GCA) Analyzer.

This module provides functionality for analyzing group conversations using the
GCA framework. It includes components for:
- Participant interaction analysis
- Metrics calculation (participation, responsivity, cohesion)
- Content analysis using LLM-based text processing
- Visualization and statistical reporting

The analyzer implements metrics and algorithms described in the GCA paper,
with support for multiple languages through advanced LLM-based processing.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import Config, default_config
from .llm_processor import LLMTextProcessor
from .logger import logger
from .utils import cosine_similarity_matrix, measure_time


class GCAAnalyzer:
    """Main analyzer class for group conversation analysis.

    This class integrates text processing, metrics calculation, and visualization
    components to provide comprehensive analysis of group conversations.
    It supports multiple languages through advanced LLM-based text processing
    and implements various metrics from the GCA framework.

    Attributes:
        _config: Configuration instance containing analysis parameters.
        llm_processor: LLM processor instance for text analysis.
    """

    def __init__(
        self,
        llm_processor: Optional[LLMTextProcessor] = None,
        config: Optional[Config] = None
    ) -> None:
        """Initialize the GCA Analyzer with required components.

        Args:
            llm_processor: LLM processor instance for text analysis.
                If None, creates a new instance using config parameters.
            config: Configuration instance with analysis parameters.
                If None, uses default configuration.
        """
        self._config = config or default_config
        self.llm_processor = llm_processor or LLMTextProcessor(
            model_name=self._config.model.model_name,
            mirror_url=self._config.model.mirror_url
        )
        logger.info("Initializing GCA Analyzer")
        logger.info("Using LLM-based text processor for multi-language support")
        logger.debug("Components initialized successfully")

    @measure_time("participant_pre")
    def participant_pre(
        self,
        conversation_id: str,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], List[int], int, int, pd.DataFrame]:
        """Preprocess participant data for analysis.

        This function filters and validates conversation data, sorts by timestamp,
        and creates a participation matrix for further analysis.

        Args:
            conversation_id: Unique identifier for the conversation.
            data: DataFrame containing participant data with required columns:
                - conversation_id: Conversation identifier
                - person_id: Participant identifier
                - time: Message timestamp
                - text: Message content

        Returns:
            A tuple containing:
                - current_data: Preprocessed DataFrame for the conversation
                - person_list: List of unique participant IDs
                - seq_list: List of message sequence numbers
                - k: Number of participants
                - n: Number of messages
                - M: Participation matrix (participants x sequences)

        Raises:
            ValueError: If no data found for conversation_id or missing required columns.
        """
        # Filter data for current conversation
        current_data = data[data.conversation_id == conversation_id].copy()
        
        if current_data.empty:  # pragma: no cover
            raise ValueError(f"No data found for conversation_id: {conversation_id}")
            
        # Validate required columns
        required_columns = ['conversation_id', 'person_id', 'time', 'text']
        missing_columns = [
            col for col in required_columns if col not in current_data.columns
        ]
        if missing_columns:  # pragma: no cover
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Sort by timestamp and assign sequence numbers
        current_data['parsed_time'] = pd.to_datetime(
            current_data['time'], format='mixed'
        )
        current_data = current_data.sort_values('parsed_time').reset_index(drop=True)
        current_data['seq_num'] = range(1, len(current_data) + 1)
        
        # Extract unique participants and sequences
        person_list = sorted(current_data.person_id.unique())
        seq_list = sorted(current_data.seq_num.unique())
        
        k = len(person_list)  # Number of participants
        n = len(seq_list)     # Number of messages
        
        # Create participation matrix
        M = pd.DataFrame(0, index=person_list, columns=seq_list)
        for _, row in current_data.iterrows():
            M.loc[row.person_id, row.seq_num] = 1
        
        return current_data, person_list, seq_list, k, n, M

    @measure_time("find_best_window_size")
    def find_best_window_size(
        self,
        data: pd.DataFrame,
        best_window_indices: Optional[float] = None,
        min_num: Optional[int] = None,
        max_num: Optional[int] = None
    ) -> int:
        """Find the optimal window size for conversation analysis.

        This function determines the best window size that satisfies the
        participation threshold criteria. It iteratively tests window sizes
        from min_num to max_num until finding one where the participation
        rate meets or exceeds the target threshold.

        Args:
            data: Input DataFrame containing conversation data.
            best_window_indices: Target participation threshold (0 to 1).
                If None, uses value from config.
            min_num: Minimum window size to consider.
                If None, uses value from config.
            max_num: Maximum window size to consider.
                If None, uses value from config or data length.

        Returns:
            int: Optimal window size that meets participation criteria.

        Raises:
            ValueError: If min_num > max_num or best_window_indices not in [0,1].
        """
        best_window_indices = (
            best_window_indices or self._config.window.best_window_indices
        )
        min_num = min_num or self._config.window.min_window_size
        max_num = max_num or self._config.window.max_window_size or len(data)

        if min_num > max_num:
            raise ValueError("min_num cannot be greater than max_num")

        if not 0 <= best_window_indices <= 1:
            raise ValueError("best_window_indices must be between 0 and 1")

        if best_window_indices == 0:  # pragma: no cover
            return min_num
        if best_window_indices == 1:  # pragma: no cover
            return max_num

        n = len(data)
        person_contributions = data.groupby('person_id')

        # Test each window size
        for w in range(min_num, max_num + 1):
            for t in range(n - w + 1):
                window_data = data.iloc[t:t+w]
                window_counts = window_data.groupby('person_id').size()
                active_participants = (
                    window_counts >= self._config.window.act_participant_indices
                ).sum()
                total_participants = len(person_contributions)
                participation_rate = active_participants / total_participants

                if participation_rate >= best_window_indices:
                    logger.info(
                        f"Found valid window size: {w} "
                        f"(threshold: {best_window_indices})"
                    )
                    return w

        # If no valid window size found, use max_num
        logger.warning(
            f"No valid window size found between {min_num} and {max_num}, "
            f"using max_num: {max_num} (threshold: {best_window_indices})"
        )   # pragma: no cover
        return max_num  # pragma: no cover

    @measure_time("calculate_participation_metrics")
    def calculate_participation_metrics(
        self,
        metrics_df: pd.DataFrame,
        M: pd.DataFrame,
        person_list: List[str],
        seq_list: List[int],
        n: int,
        k: int
    ) -> pd.DataFrame:
        """Calculate participation-related metrics for each participant.

        Computes the following metrics based on the GCA framework:
        - Pa: Raw participation count (Formula 4)
        - Pa_average: Average participation rate (Formula 5)
        - Pa_std: Standard deviation of participation (Formula 6)
        - Pa_hat: Normalized participation rate (Formula 9)

        Args:
            metrics_df: DataFrame to store the calculated metrics.
            M: Participation matrix (participants x sequences).
            person_list: List of participant IDs.
            seq_list: List of sequence numbers.
            n: Number of messages.
            k: Number of participants.

        Returns:
            DataFrame with updated participation metrics for each participant.
        """
        # Calculate participation metrics (Formula 4 and 5)
        for person in person_list:
            # Pa = sum(M_a) (Formula 4)
            metrics_df.loc[person, 'Pa'] = M.loc[person].sum()
            # p̄a = (1/n)||Pa|| (Formula 5)
            metrics_df.loc[person, 'Pa_average'] = metrics_df.loc[person, 'Pa'] / n
            
        # Calculate participation standard deviation (Formula 6)
        for person in person_list:
            variance = sum(
                (M.loc[person, seq] - metrics_df.loc[person, 'Pa_average'])**2
                for seq in seq_list
            )
            metrics_df.loc[person, 'Pa_std'] = np.sqrt(variance / (n-1))
            
        # Calculate relative participation (Formula 9)
        metrics_df['Pa_hat'] = (
            metrics_df['Pa_average'] - 1/k
        ) / (1/k)
            
        return metrics_df

    @measure_time("get_Ksi_lag")
    def get_Ksi_lag(
        self,
        best_window_length: int,
        person_list: List[str],
        k: int,
        seq_list: List[int],
        M: pd.DataFrame,
        cosine_similarity_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the Ksi lag matrix for interaction analysis.

        Computes the w-spanning cross-cohesion matrix (Ξ) using Formula 16
        from the GCA framework. This matrix captures the semantic and temporal
        relationships between participants' contributions.

        Args:
            best_window_length: Optimal window size (w) for analysis.
            person_list: List of participant IDs.
            k: Number of participants.
            seq_list: List of contribution sequence numbers.
            M: Participation matrix (participants x sequences).
            cosine_similarity_matrix: Matrix of semantic similarities between messages.

        Returns:
            pd.DataFrame: Ksi lag matrix (R_w) containing cross-cohesion values
                between all pairs of participants.
        """
        # Initialize w-spanning cross-cohesion matrix with float dtype
        X_tau = pd.DataFrame(0.0, index=person_list, columns=person_list, dtype=float)
        w = best_window_length
        
        # Convert seq_list to sorted numpy array for faster operations
        sorted_seqs = np.array(sorted(seq_list))
        
        # Pre-compute all possible lagged indices for each tau
        lag_indices = {
            tau: np.arange(tau, len(sorted_seqs))
            for tau in range(1, w + 1)
        }
            
        # Convert matrices to numpy arrays for faster operations
        M_np = M.loc[:, sorted_seqs].to_numpy()
        cos_sim_np = cosine_similarity_matrix.loc[sorted_seqs, sorted_seqs].to_numpy()
        
        # Calculate cross-cohesion for each tau and accumulate
        for tau in range(1, w + 1):
            indices = lag_indices[tau]
            lagged_indices = indices - tau
            
            for a_idx, a in enumerate(person_list):
                for b_idx, b in enumerate(person_list):
                    # Vectorized calculation of Pab_tau
                    Pab_tau = np.sum(
                        M_np[a_idx, lagged_indices] * M_np[b_idx, indices]
                    )
                    
                    if Pab_tau > 0:
                        # Vectorized calculation of Sab_sum
                        Sab_sum = np.sum(
                            M_np[a_idx, lagged_indices] * 
                            M_np[b_idx, indices] * 
                            cos_sim_np[lagged_indices, indices]
                        )
                        X_tau.loc[a, b] += Sab_sum / Pab_tau    # TODO: Add copresence and directly reply two modes use 'person_id, to_person_id'
        
        # Formula 17: Responsivity across w
        R_w = X_tau.multiply(1.0/w)
        
        return R_w

    @measure_time("calculate_cohesion_response")
    def calculate_cohesion_response(
        self,
        person_list: List[str],
        k: int,
        R_w: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate cohesion and response metrics for each participant.

        Computes three key interaction metrics from the GCA framework:
        - Internal cohesion (Ca): Self-response coherence (Formula 18)
        - Overall responsivity (Ra): Response behavior to others (Formula 19)
        - Social impact (Ia): Impact on others' responses (Formula 20)

        Args:
            person_list: List of participant IDs.
            k: Number of participants.
            R_w: Responsivity matrix across window w.

        Returns:
            Tuple containing three pd.Series:
            - Internal cohesion values for each participant
            - Overall responsivity values for each participant
            - Social impact values for each participant
        """
        metrics_df = pd.DataFrame(
            index=person_list,
            columns=['Internal_cohesion', 'Overall_responsivity', 'Social_impact'],
            dtype=float
        )

        for person in person_list:
            # Calculate Internal cohesion with w-spanning (Formula 18)
            metrics_df.loc[person, 'Internal_cohesion'] = R_w.loc[person, person]
        
            # Calculate Overall responsivity with w-spanning (Formula 19)
            metrics_df.loc[person, 'Overall_responsivity'] = sum(
                R_w.loc[person, other] for other in person_list if other != person
            ) / (k-1)
        
            # Calculate Social impact with w-spanning (Formula 20)
            metrics_df.loc[person, 'Social_impact'] = sum(
                R_w.loc[other, person] for other in person_list if other != person
            ) / (k-1)

        return (
            metrics_df['Internal_cohesion'],
            metrics_df['Overall_responsivity'],
            metrics_df['Social_impact']
        )

    @measure_time("calculate_personal_given_new_dict")
    def calculate_personal_given_new_dict(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        current_data: pd.DataFrame
    ) -> Tuple[dict, dict]:
        """Calculate LSA metrics for all contributions and aggregate by participant.

        Processes each contribution to compute:
        - Newness (n_c_t): Proportion of new content (Formula 25)
        - Communication density (D_i): Information density (Formula 27)

        The metrics are first calculated for each contribution, then grouped
        by participant for further analysis.

        Args:
            vectors: List of document vectors from LSA processing.
            texts: List of corresponding message texts.
            current_data: DataFrame containing participant and message data.

        Returns:
            Tuple containing two dictionaries:
            - n_c_t_dict: Mapping of participant ID to list of newness values
            - D_i_dict: Mapping of participant ID to list of density values
        """
        # First calculate metrics for each text
        current_data = self.calculate_text_given_new_df(vectors, texts, current_data)
        
        # Group by person_id and convert to dictionaries
        grouped = current_data.groupby('person_id')
        n_c_t_dict = {
            person: list(group['n_c_t'])
            for person, group in grouped
        }
        D_i_dict = {
            person: list(group['D_i'])
            for person, group in grouped
        }
        
        return n_c_t_dict, D_i_dict

    @measure_time("calculate_personal_given_new_averages")
    def calculate_personal_given_new_averages(
        self,
        person_list: List[str],
        n_c_t_dict: dict,
        D_i_dict: dict
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate average LSA metrics for each participant.

        Computes the average newness (Formula 26) and communication density
        (Formula 28) for each participant based on their individual message
        metrics.

        Args:
            person_list: List of participant IDs.
            n_c_t_dict: Dictionary mapping participant IDs to newness values.
            D_i_dict: Dictionary mapping participant IDs to density values.

        Returns:
            Tuple containing two pd.Series:
            - Average newness values for each participant
            - Average communication density values for each participant
        """
        metrics_df = pd.DataFrame(
            0.0,
            index=person_list,
            columns=['Newness', 'Communication_density'],
            dtype=float
        )

        for person in person_list:
            if person in n_c_t_dict and len(n_c_t_dict[person]) > 0:
                # Formula 26: Average newness
                metrics_df.loc[person, 'Newness'] = np.nanmean(
                    n_c_t_dict[person]
                ) if len(n_c_t_dict[person]) > 0 else 0.0
                # Formula 28: Average communication density
                metrics_df.loc[person, 'Communication_density'] = np.nanmean(
                    D_i_dict[person]
                ) if len(D_i_dict[person]) > 0 else 0.0
            else:
                metrics_df.loc[person, 'Newness'] = 0.0
                metrics_df.loc[person, 'Communication_density'] = 0.0
                
        return (
            metrics_df['Newness'],
            metrics_df['Communication_density']
        )

    @measure_time("calculate_text_given_new_df")
    def calculate_text_given_new_df(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        current_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate LSA metrics for all contributions using batch processing.

        For each message in the conversation, computes:
        - Newness (n_c_t): Proportion of new content (Formula 25)
        - Communication density (D_i): Information density (Formula 27)

        Args:
            vectors: List of document vectors from LSA processing.
            texts: List of corresponding message texts.
            current_data: DataFrame containing conversation data.

        Returns:
            DataFrame with original data plus two new columns:
            - n_c_t: Newness values for each message
            - D_i: Communication density values for each message
        """
        results = []
        for idx in range(len(vectors)):
            if idx == 0:
                n_c_t = 1.0  # First message is entirely new
            else:
                n_c_t = self._calculate_newness_proportion(vectors, idx)    # TODO: Add topic clustering to weight newness
            # Convert DataFrame row to numpy array for density calculation
            vector = vectors.iloc[idx].to_numpy() if isinstance(vectors, pd.DataFrame) else vectors[idx]
            D_i = self._calculate_communication_density(vector, texts[idx])
            results.append((n_c_t, D_i))
        
        return pd.concat([
            current_data,
            pd.DataFrame(results, columns=['n_c_t', 'D_i'])
        ], axis=1)

    @measure_time("calculate_batch_lsa_metrics")
    def _calculate_batch_lsa_metrics(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        start_idx: int,
        end_idx: int
    ) -> List[Tuple[float, float]]:
        """Calculate LSA metrics for a batch of contributions.

        Processes a subset of messages from start_idx to end_idx to compute
        their newness and communication density metrics. This method enables
        efficient batch processing of large conversations.

        Args:
            vectors: List of document vectors from LSA processing.
            texts: List of corresponding message texts.
            start_idx: Starting index in the batch.
            end_idx: Ending index in the batch (exclusive).

        Returns:
            List of tuples, each containing:
            - float: Newness value (n_c_t) for the message
            - float: Communication density value (D_i) for the message
        """
        results = []
        for idx in range(start_idx, end_idx):
            if idx == 0:
                n_c_t = 1.0  # First message is entirely new
            else:
                n_c_t = self._calculate_newness_proportion(vectors, idx)
            # Convert DataFrame row to numpy array for density calculation
            vector = vectors.iloc[idx].to_numpy() if isinstance(vectors, pd.DataFrame) else vectors[idx]
            D_i = self._calculate_communication_density(vector, texts[idx])
            results.append((n_c_t, D_i))
        return results

    def _calculate_newness_proportion(
        self,
        vectors: List[np.ndarray],
        current_idx: int
    ) -> float:
        """Calculate the proportion of new content in the current contribution.

        Implements Formula 25 from the GCA framework to compute the proportion
        of new content in a message relative to all previous messages. Uses
        orthogonal projection to separate new content from previously seen content.

        Args:
            vectors: List of document vectors for all messages.
            current_idx: Index of the current message to analyze.

        Returns:
            float: Proportion of new content (n_c_t), in range [0, 1].
                1.0 indicates entirely new content,
                0.0 indicates no new content.
        """
        if current_idx == 0:  # pragma: no cover
            return 1.0  # First message is entirely new

        # Convert previous vectors to numpy arrays if needed
        if isinstance(vectors, pd.DataFrame):
            prev_vectors = vectors.iloc[:current_idx].to_numpy()
            d_i = vectors.iloc[current_idx].to_numpy()
        else:
            prev_vectors = [v.values if isinstance(v, pd.Series) else v for v in vectors[:current_idx]]
            prev_vectors = np.vstack(prev_vectors)
            d_i = vectors[current_idx].values if isinstance(vectors[current_idx], pd.Series) else vectors[current_idx]

        # Calculate projection matrix efficiently using SVD
        U, _, _ = np.linalg.svd(prev_vectors.T, full_matrices=False)
        g_i = U @ (U.T @ d_i)  # Project onto previous content space
        n_i = d_i - g_i        # Extract new content component

        # Calculate proportion of new content
        n_norm = np.linalg.norm(n_i)
        g_norm = np.linalg.norm(g_i)
        
        return n_norm / (n_norm + g_norm) if (n_norm + g_norm) > 0 else 0.0

    def _calculate_communication_density(
        self,
        vector: np.ndarray,
        text: str
    ) -> float:
        """Calculate the communication density of a contribution.

        Implements Formula 27 from the GCA framework to compute the density
        of information in a message. This is calculated as the magnitude of
        the message's vector representation divided by its text length.

        Args:
            vector: Document vector representation of the message.
            text: Original text content of the message.

        Returns:
            float: Communication density (D_i).
                Higher values indicate more information per character.
                Returns 0.0 for empty messages.
        """
        text_length = len(text)
        if text_length == 0:
            return 0.0
            
        # Convert vector to numpy array if it's a pandas Series
        if isinstance(vector, pd.Series):
            vector = vector.values
            
        return np.linalg.norm(vector) / text_length

    @measure_time("analyze_conversation")
    def analyze_conversation(
        self,
        conversation_id: str,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze a conversation's dynamics using GCA metrics.

        This is the main analysis method that computes all GCA metrics for a
        given conversation. The following metrics are calculated according to
        the formulas in the GCA framework:

        Participation Metrics:
        - Pa: Raw participation count (Formula 4)
        - p̄a: Average participation rate (Formula 5)
        - σa: Standard deviation of participation (Formula 6)
        - P̂a: Normalized participation rate (Formula 9)

        Interaction Metrics:
        - Ξ: Cross-cohesion matrix (Formula 16)
        - Ca: Internal cohesion (Formula 18)
        - Ra: Overall responsivity (Formula 19)
        - Ia: Social impact (Formula 20)

        Content Metrics:
        - n(ct): Message newness (Formula 25)
        - Di: Communication density (Formula 27)

        Args:
            conversation_id: Unique identifier for the conversation.
            data: DataFrame containing conversation data with required columns:
                - conversation_id: Conversation identifier
                - person_id: Participant identifier
                - text: Message content
                - timestamp: Message timestamp
                - seq: Message sequence number

        Returns:
            DataFrame containing calculated GCA metrics for each participant:
                - conversation_id: Input conversation identifier
                - participation: Normalized participation rate
                - responsivity: Response behavior to others
                - internal_cohesion: Self-response coherence
                - social_impact: Impact on others' responses
                - newness: Average message novelty
                - comm_density: Average message information density
        """
        # Preprocess conversation data
        current_data, person_list, seq_list, k, n, M = self.participant_pre(
            conversation_id, data
        )

        # Initialize metrics DataFrame
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
        
        # Calculate participation metrics
        metrics_df = self.calculate_participation_metrics(
            metrics_df=metrics_df,
            M=M,
            person_list=person_list,
            seq_list=seq_list,
            n=n,
            k=k
        )
        
        # Process text content
        texts = current_data.text.to_list()
        vectors = self.llm_processor.doc2vector(texts)
        
        # Find optimal window size
        w = self.find_best_window_size(current_data)
        print(f"=== Found valid window size: {w} ===")
        print(f"=== (best window threshold: {self._config.window.best_window_indices}) ===")
        print(f"=== (each participant in a window that is greater than or equal to: {self._config.window.act_participant_indices}) ===")
        
        # Calculate semantic similarity
        cosine_matrix = cosine_similarity_matrix(
            vectors, seq_list, current_data
        )
        
        # Calculate interaction metrics
        R_w = self.get_Ksi_lag(
            w, person_list, k, seq_list, M, cosine_matrix
        )
        
        # Calculate cohesion and response metrics
        metrics_df['Internal_cohesion'], metrics_df['Overall_responsivity'], metrics_df['Social_impact'] = \
            self.calculate_cohesion_response(
                person_list=person_list,
                k=k,
                R_w=R_w
            )
        
        # Calculate content metrics
        n_c_t_dict, D_i_dict = self.calculate_personal_given_new_dict(
            vectors=vectors,
            texts=texts,
            current_data=current_data
        )
        
        # Calculate average content metrics
        metrics_df['Newness'], metrics_df['Communication_density'] = \
            self.calculate_personal_given_new_averages(
                person_list=person_list,
                n_c_t_dict=n_c_t_dict,
                D_i_dict=D_i_dict
            )
        
        # Rename columns to match paper terminology
        metrics_df = metrics_df.rename(columns={
            'Pa_hat': 'participation',
            'Overall_responsivity': 'responsivity',
            'Internal_cohesion': 'internal_cohesion',
            'Social_impact': 'social_impact',
            'Newness': 'newness',
            'Communication_density': 'comm_density'
        })
        
        return metrics_df

    @measure_time("calculate_descriptive_statistics")
    def calculate_descriptive_statistics(
        self,
        all_metrics: dict,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate descriptive statistics for GCA measures.

        Computes comprehensive descriptive statistics for all GCA metrics
        across multiple conversations. Statistics include:
        - Central tendency measures (mean, median)
        - Dispersion measures (standard deviation, coefficient of variation)
        - Data quality indicators (missing values count)

        Args:
            all_metrics: Dictionary mapping conversation IDs to DataFrames
                containing GCA metrics for each participant.
            output_dir: Optional directory path to save statistics CSV file.
                If None, statistics will only be displayed.

        Returns:
            DataFrame containing the following statistics for each measure:
                - Minimum: Minimum value
                - Median: Median value
                - Mean: Mean value 
                - SD: Standard deviation
                - Maximum: Maximum value
                - Count: Number of non-null values
                - Missing: Number of null values
                - CV: Coefficient of variation (SD divided by absolute Mean)

        Note:
            - All numeric values are rounded to 2 decimal places
            - CV is set to infinity when mean is zero or when mean/SD is null
            - Results are both printed in a formatted table and returned
        """
        all_data = pd.concat(all_metrics.values())
        
        # Calculate basic statistics
        stats = pd.DataFrame({
            'Minimum': all_data.min(),
            'Median': all_data.median(),
            'Mean': all_data.mean(),
            'SD': all_data.std(),
            'Maximum': all_data.max(),
            'Count': all_data.count(),
            'Missing': all_data.isnull().sum()
        })
        
        # Calculate CV with handling for division by zero
        means = all_data.mean()
        stds = all_data.std()
        cvs = pd.Series(index=means.index, dtype=float)
        
        for col in means.index:
            mean = means[col]
            std = stds[col]
            if mean == 0 or pd.isna(mean) or pd.isna(std):
                cvs[col] = float('inf')
            else:
                cvs[col] = std / abs(mean)  # Use absolute mean for CV
        
        stats['CV'] = cvs
        stats = stats.round(2)
        
        # Print formatted statistics table
        print("=== Descriptive statistics for GCA measures ===")
        print("-" * 80)
        print(
            "Measure".ljust(20),
            "Minimum  Median  Mean    SD     Maximum  Count  Missing  CV"
        )
        print("-" * 80)
        
        for measure in stats.index:
            row = stats.loc[measure]
            cv_value = f"{row['CV']:.2f}" if row['CV'] < 10 else 'inf'
            print(
                f"{measure.replace('_', ' ').title().ljust(20)}"
                f"{row['Minimum']:7.2f}  "
                f"{row['Median']:6.2f}  "
                f"{row['Mean']:5.2f}  "
                f"{row['SD']:5.2f}  "
                f"{row['Maximum']:7.2f}  "
                f"{row['Count']:5.0f}  "
                f"{row['Missing']:7.0f}  "
                f"{cv_value:>5}"
            )
        print("-" * 80)
        
        # Save statistics to file if output directory provided
        if output_dir:  # pragma: no cover
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'descriptive_statistics_gca.csv')
            stats.to_csv(output_file)
            print(f"=== Saved descriptive statistics to: {output_file} ===")
        
        return stats