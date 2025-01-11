import pandas as pd
from typing import Tuple, List, Dict, Any
from .text_processor import TextProcessor
from .metrics import MetricsCalculator
from .logger import logger
import numpy as np

class GCAAnalyzer:
    def __init__(self):
        """Initialize the GCA Analyzer with required components."""
        logger.info("Initializing GCA Analyzer")
        self.text_processor = TextProcessor()
        self.metrics = MetricsCalculator()
        logger.debug("Components initialized successfully")

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
            for t in range(len(time_list)):
                window_end = t + w
                if window_end > len(time_list):
                    break
                    
                # Get the actual time values for the window
                window_times = time_list[t:window_end]
                current_user_data = pd.DataFrame()
                current_user_data['temp'] = M[window_times].apply(lambda x: x.sum(), axis=1)
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
        根据论文公式分析视频对话数据。
        
        Args:
            video_id: 视频ID
            data: 包含对话数据的DataFrame
            
        Returns:
            pd.DataFrame: 每个参与者的分析结果
        """
        # 预处理数据
        current_data, person_list, time_list, k, n, M = self.participant_pre(video_id, data)
        
        # 初始化结果DataFrame
        student = pd.DataFrame(0.0, 
                             index=person_list,
                             columns=['Pa', 'Pa_average', 'Pa_std', 'video_id', 'Pa_hat',
                                    'Internal_cohesion', 'Overall_responsivity',
                                    'Social_impact', 'Newness', 'Communication_density'],
                             dtype=float)
        
        student['video_id'] = video_id
        
        # 计算参与度指标 (根据公式4和5)
        for person in person_list:
            # ||Pa|| = Σ pa(t) (公式4)
            student.loc[person, 'Pa'] = M.loc[person].sum()
            # p̄a = (1/n)||Pa|| (公式5)
            student.loc[person, 'Pa_average'] = student.loc[person, 'Pa'] / n
            
        # 计算参与度标准差 (公式6)
        for person in person_list:
            variance = 0
            for t in time_list:
                variance += (M.loc[person, t] - student.loc[person, 'Pa_average'])**2
            student.loc[person, 'Pa_std'] = np.sqrt(variance / (n-1))
            
        # 计算相对参与度 (公式9的修正版)
        student['Pa_hat'] = (student['Pa_average'] - 1/k) / (1/k)
        
        # 处理文本并计算向量
        vector, dataset = self.text_processor.doc2vector(current_data.text_clean)
        
        # 计算余弦相似度矩阵
        cosine_similarity_matrix = pd.DataFrame(np.zeros((len(time_list), len(time_list)), dtype=float), index=time_list, columns=time_list)
        for i in range(len(vector)):
            for j in range(len(vector)):
                cosine_similarity_matrix.iloc[i, j] = self.metrics.cosine_similarity(vector[i], vector[j])
        
        # 获取最佳窗口大小
        w = self.get_best_window_num(time_list, M)
        
        # 计算Cross-cohesion (公式15和16)
        cross_cohesion = pd.DataFrame(0.0, index=person_list, columns=person_list)
        for a in person_list:
            for b in person_list:
                for tau in range(1, w+1):
                    Pab_tau = 0
                    Sab_sum = 0
                    for t in range(tau, n):
                        # ||Pab(τ)|| (公式16)
                        Pab_tau += M.loc[a, time_list[t-tau]] * M.loc[b, time_list[t]]
                        if Pab_tau > 0:
                            # ξab(τ) (公式15)
                            Sab_sum += M.loc[a, time_list[t-tau]] * M.loc[b, time_list[t]] * \
                                     cosine_similarity_matrix.iloc[t-tau, t]
                    if Pab_tau > 0:
                        cross_cohesion.loc[a, b] += Sab_sum / Pab_tau
        cross_cohesion = cross_cohesion / w
        
        # 计算Overall responsivity (公式19)
        for person in person_list:
            responsivity_sum = 0
            for other in person_list:
                if other != person:
                    responsivity_sum += cross_cohesion.loc[person, other]
            student.loc[person, 'Overall_responsivity'] = responsivity_sum / (k-1)
        
        # 计算Internal cohesion (公式18)
        for person in person_list:
            student.loc[person, 'Internal_cohesion'] = cross_cohesion.loc[person, person]
        
        # 计算Social impact (公式20)
        for person in person_list:
            impact_sum = 0
            for other in person_list:
                if other != person:
                    impact_sum += cross_cohesion.loc[other, person]
            student.loc[person, 'Social_impact'] = impact_sum / (k-1)
        
        # 计算Newness (公式25和26)
        for person in person_list:
            person_messages = current_data[current_data.person_id == person]
            if not person_messages.empty:
                newness_sum = 0
                for idx, row in person_messages.iterrows():
                    # 获取之前的所有消息向量
                    historical_vectors = [vector[i] for i in range(idx)]
                    if historical_vectors:
                        try:
                            # 计算正交投影
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
                                # 使用QR分解计算正交补空间的投影
                                Q, R = np.linalg.qr(historical_matrix.T)
                                proj = current_vector - Q @ (Q.T @ current_vector)
                                # n(ct) (公式25)
                                newness = np.linalg.norm(proj) / (np.linalg.norm(proj) + np.linalg.norm(current_vector))
                                newness_sum += newness
                        except (ValueError, np.linalg.LinAlgError) as e:
                            # If we encounter any linear algebra errors, skip this vector
                            continue
                # Na (公式26)
                student.loc[person, 'Newness'] = newness_sum / student.loc[person, 'Pa']
                
        # 计算Communication density (公式27和28)
        for person in person_list:
            person_messages = current_data[current_data.person_id == person]
            if not person_messages.empty:
                density_sum = 0
                for idx, row in person_messages.iterrows():
                    # Di (公式27)
                    vector_norm = np.linalg.norm([v[1] for v in vector[idx]])
                    word_length = len(dataset[idx])
                    if word_length > 0:
                        density_sum += vector_norm / word_length
                # D̄a (公式28)
                student.loc[person, 'Communication_density'] = density_sum / student.loc[person, 'Pa']
                
        return student
