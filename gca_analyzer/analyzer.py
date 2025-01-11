import pandas as pd
from typing import Tuple, List, Dict, Any
from .text_processor import TextProcessor
from .metrics import MetricsCalculator

class GCAAnalyzer:
    def __init__(self):
        """Initialize the GCA Analyzer with required components."""
        self.text_processor = TextProcessor()
        self.metrics = MetricsCalculator()

    def participant_pre(self, video_id: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[int], int, int, pd.DataFrame]:
        """
        Preprocess participant data.
        
        Args:
            video_id: ID of the video to analyze
            data: DataFrame containing all participant data
            
        Returns:
            tuple: (current_data, person_list, time_list, k, n, M)
                  where M is a participation matrix (DataFrame) with shape (k, n)
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
            
        # Clean text data
        current_data['text_clean'] = current_data.text.apply(self.text_processor.chinese_word_cut)
        
        # Get unique participants and timestamps
        person_list = current_data.person_id.unique().tolist()
        time_list = sorted(current_data.time.unique().tolist())
        
        # Calculate dimensions
        k = len(person_list)  # Number of participants
        n = len(time_list)    # Number of time points
        
        # Create participation matrix M
        M = pd.DataFrame(0, index=person_list, columns=time_list)
        for _, row in current_data.iterrows():
            M.loc[row.person_id, row.time] = 1
        
        return current_data, person_list, time_list, k, n, M

    def get_best_window_num(self, time_list: List[int], M: pd.DataFrame, best_window_indices: float = 0.3, min_num: int = 2, max_num: int = 10) -> int:
        """
        Find the optimal window size for analysis.
        
        Args:
            time_list: List of time points
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
            
        n = len(time_list)
        for w in range(min_num, max_num):
            found_valid_window = False
            for t in time_list:
                window_end = t + w
                if window_end > n:
                    break
                    
                col_list = list(range(t, window_end))
                current_user_data = pd.DataFrame()
                current_user_data['temp'] = M[col_list].apply(lambda x: x.sum(), axis=1)
                percen = len(current_user_data[current_user_data['temp'] >= 2]) / len(current_user_data['temp'])
                
                if percen >= best_window_indices:
                    found_valid_window = True
                    break
                    
            if found_valid_window:
                return w
        return max_num

    def get_Ksi_lag(self, best_window_length: int, person_list: List[str], k: int,
                    time_list: List[int], M: pd.DataFrame, cosine_similarity_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Ksi lag matrix for interaction analysis.
        
        Args:
            best_window_length: Optimal window size
            person_list: List of participants
            k: Number of participants
            time_list: List of time points
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
                    
                    for t in time_list:
                        if t < tao + 1:
                            continue
                        if t != time_list[-1]:
                            Pab_tao += M.loc[a, t-tao] * M.loc[b, t]
                        else:
                            if Pab_tao == 0:
                                _Ksi_lag.loc[a, b] = Pab_tao
                            else:
                                Sabtu = sum(M.loc[a, t - tao] * M.loc[b, t] * 
                                          cosine_similarity_matrix.loc[t - tao, t] 
                                          for t in time_list if t >= tao + 1)
                                _Ksi_lag.loc[a, b] = Sabtu / Pab_tao
                                
            Ksi_lag += _Ksi_lag
            
        return Ksi_lag * (1/w)

    def analyze_video(self, video_id: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze a video's group conversation.
        
        Args:
            video_id: The ID of the video to analyze
            data: DataFrame containing conversation data
            
        Returns:
            pd.DataFrame: Analysis results for each participant
        """
        # Preprocess data
        current_data, person_list, time_list, k, n, M = self.participant_pre(video_id, data)
        
        # Initialize results DataFrame with float type
        student = pd.DataFrame(0.0, 
                             index=person_list,
                             columns=['Pa', 'Pa_average', 'Pa_std', 'video_id', 'Pa_hat',
                                    'Internal_cohesion', 'Overall_responsivity',
                                    'Social_impact', 'Newness', 'Communication_density'],
                             dtype=float)
        
        # Set video_id
        student['video_id'] = video_id
        
        # Calculate participation metrics
        for person in person_list:
            student.loc[person, 'Pa'] = M.loc[person].sum() / M.sum().sum()
            
        student['Pa_average'] = student['Pa'].mean()
        student['Pa_std'] = student['Pa'].std()
        student['Pa_hat'] = student['Pa'] / student['Pa_average']
        
        # Process text and calculate vectors
        vector, dataset = self.text_processor.doc2vector(current_data.text_clean)
        
        # Calculate interaction metrics
        _Ksi_lag = pd.DataFrame(0.0, index=person_list, columns=person_list, dtype=float)
        
        for t in range(1, n):
            for a in person_list:
                for b in person_list:
                    if a != b:
                        # Calculate Sab(t,u)
                        Sabtu = 0
                        Pab_tao = 0
                        
                        for u in range(1, t+1):
                            if (current_data.person_id.iloc[t-1] == a and 
                                current_data.person_id.iloc[u-1] == b):
                                Sabtu = self.metrics.cosine_similarity(
                                    vector[t-1], vector[u-1])
                                Pab_tao = 1
                                break
                                
                        if Pab_tao > 0:
                            _Ksi_lag.loc[a, b] = Sabtu / Pab_tao
                            
        # Calculate participant metrics
        for a in person_list:
            # Internal cohesion
            v = 0
            for t in range(n):
                if current_data.person_id.iloc[t] == a:
                    for u in range(t):
                        if current_data.person_id.iloc[u] == a:
                            v += self.metrics.cosine_similarity(vector[t], vector[u])
            student.loc[a, 'Internal_cohesion'] = v / (n * n) if n > 0 else 0
            
            # Overall responsivity and social impact
            for b in person_list:
                if a != b:
                    v = _Ksi_lag.loc[a, b]
                    student.loc[a, 'Overall_responsivity'] += v
                    student.loc[b, 'Social_impact'] += v
                    
        # Normalize metrics
        n_others = len(person_list) - 1
        if n_others > 0:
            student['Overall_responsivity'] /= n_others
            student['Social_impact'] /= n_others
            
        # Calculate newness for each participant
        for person in person_list:
            person_indices = current_data.index[current_data.person_id == person].tolist()
            if person_indices:
                # Get all messages before this person's first message
                historical_indices = list(range(person_indices[0]))
                if historical_indices:
                    # Get all historical messages
                    historical_vectors = [vector[i] for i in historical_indices]
                    
                    # Get this person's first message
                    current_idx = person_indices[0]
                    
                    # Calculate newness based on similarity to historical messages
                    student.loc[person, 'Newness'] = self.metrics.compute_nct(
                        historical_vectors, vector[current_idx])
                else:
                    # First message in the conversation gets maximum newness
                    student.loc[person, 'Newness'] = 1.0
                    
        # Calculate communication density
        for person in person_list:
            person_data = current_data[current_data.person_id == person]
            if not person_data.empty:
                idx = person_data.index[0]
                student.loc[person, 'Communication_density'] = self.metrics.compute_Di(
                    vector[idx], dataset[idx])
                    
        return student
