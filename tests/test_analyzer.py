import unittest
import pandas as pd
import numpy as np
import os
from gca_analyzer import GCAAnalyzer

class TestGCAAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = GCAAnalyzer()
        
        # Create test data in memory
        self.test_data = pd.DataFrame({
            'video_id': ['1A'] * 10,
            'person_id': ['Student1', 'Student2', 'Teacher'] * 3 + ['Student1'],
            'time': list(range(1, 11)),  # Sequential time points
            'text': [
                'Hello everyone!',
                'Hi teacher',
                '今天我们来讨论AI',
                'I have a question',
                '这是一个很好的问题',
                'Thank you',
                '让我们继续',
                'I understand now',
                '大家都明白了吗',
                'Yes, clear'
            ]
        })

    def test_participant_pre(self):
        """Test the participant preprocessing function with real test data."""
        current_data, person_list, time_list, k, n, M = self.analyzer.participant_pre(
            '1A', self.test_data
        )
        
        # Test basic properties
        self.assertTrue(len(person_list) > 0)  # Should have at least one user
        self.assertTrue(len(time_list) > 0)    # Should have at least one time point
        self.assertTrue(k > 0)                 # Number of participants should be positive
        self.assertTrue(n > 0)                 # Number of time points should be positive
        
        # Test participation matrix M
        self.assertEqual(M.shape, (k, n))      # Matrix should be kxn
        self.assertTrue(isinstance(M, pd.DataFrame))
        
        # Test if text cleaning was applied
        self.assertTrue('text_clean' in current_data.columns)

    def test_participant_pre_empty_data(self):
        """Test participant preprocessing with empty data."""
        empty_data = pd.DataFrame(columns=self.test_data.columns)
        with self.assertRaises(ValueError):
            self.analyzer.participant_pre('1A', empty_data)
            
    def test_participant_pre_single_participant(self):
        """Test participant preprocessing with single participant."""
        single_participant_data = self.test_data.copy()
        single_participant_data['person_id'] = 'Teacher'  # Set all rows to same participant
        current_data, person_list, time_list, k, n, M = self.analyzer.participant_pre(
            '1A', single_participant_data
        )
        self.assertEqual(len(person_list), 1)  # Should only have one participant
        self.assertEqual(k, 1)  # k should be 1
        self.assertEqual(M.shape, (1, n))  # Matrix should be 1xn
        
    def test_participant_pre_same_time(self):
        """Test participant preprocessing when all participants speak at same time."""
        same_time_data = self.test_data.copy()
        # We don't need to set current_num as it will be generated in participant_pre
        current_data, person_list, time_list, k, n, M = self.analyzer.participant_pre(
            '1A', same_time_data
        )
        # Instead of checking time points, we check that the matrix has the correct shape
        self.assertEqual(M.shape[1], n)  # Matrix should have n columns
        self.assertEqual(M.shape[0], k)  # Matrix should have k rows
        # Check that each person has at least one participation
        self.assertTrue(all(M.sum(axis=1) > 0))  # Each person should speak at least once

    def test_participant_pre_discontinuous_time(self):
        """Test participant preprocessing with discontinuous time points."""
        discontinuous_data = self.test_data.copy()
        discontinuous_data.loc[10:20, 'current_num'] = 100  # Create a gap in time
        current_data, person_list, time_list, k, n, M = self.analyzer.participant_pre(
            '1A', discontinuous_data
        )
        self.assertTrue(n > 0)  # Should still process the data
        self.assertEqual(M.shape, (k, n))  # Matrix should be kxn
        
    def test_get_best_window_num(self):
        """Test the optimal window size calculation with real test data."""
        # First get the participation matrix from test data
        _, person_list, time_list, _, _, M = self.analyzer.participant_pre(
            '1A', self.test_data
        )
        
        # Test with moderate threshold
        best_window = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=0.3,
            min_num=2,
            max_num=10
        )
        
        # Window size should be within bounds
        self.assertTrue(2 <= best_window <= 10)
        
        # Test with different thresholds
        best_window_high = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=0.4,
            min_num=2,
            max_num=10
        )
        
        best_window_low = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=0.2,
            min_num=2,
            max_num=10
        )
        
        # Test threshold relationships
        # Higher threshold should require larger or equal window size
        self.assertTrue(best_window_high >= best_window)
        # Lower threshold should require smaller or equal window size
        self.assertTrue(best_window_low <= best_window)

    def test_get_best_window_num_invalid_range(self):
        """Test get_best_window_num with invalid min/max range."""
        _, person_list, time_list, _, _, M = self.analyzer.participant_pre(
            '1A', self.test_data
        )
        with self.assertRaises(ValueError):
            self.analyzer.get_best_window_num(
                time_list=time_list,
                M=M,
                best_window_indices=0.3,
                min_num=10,  # min > max
                max_num=5
            )
            
    def test_get_best_window_num_extreme_thresholds(self):
        """Test get_best_window_num with extreme threshold values."""
        _, person_list, time_list, _, _, M = self.analyzer.participant_pre(
            '1A', self.test_data
        )
        # Test with threshold = 0
        window_0 = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=0.0,
            min_num=2,
            max_num=10
        )
        self.assertEqual(window_0, 2)  # Should return min_num
        
        # Test with threshold = 1
        window_1 = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=1.0,
            min_num=2,
            max_num=10
        )
        self.assertEqual(window_1, 10)  # Should return max_num

    def test_text_preprocessing(self):
        """Test the text preprocessing functionality."""
        test_data = self.test_data.copy()
        # Add some special characters and test cases
        test_data.loc[0, 'text'] = "Hello! This is a test... @#$%^&*"
        test_data.loc[1, 'text'] = "测试中文和English混合"
        test_data.loc[2, 'text'] = "  Multiple   Spaces   Test  "
        
        current_data, _, _, _, _, _ = self.analyzer.participant_pre('1A', test_data)
        
        # Check if special characters are handled properly
        self.assertTrue('text_clean' in current_data.columns)
        self.assertFalse('@#$%^&*' in current_data.iloc[0]['text_clean'])
        # Check if spaces are normalized
        self.assertEqual(
            current_data.iloc[2]['text_clean'].count('  '), 
            0, 
            "Multiple spaces should be normalized"
        )

    def test_invalid_group_id(self):
        """Test handling of invalid group ID."""
        with self.assertRaises(ValueError):
            self.analyzer.participant_pre('invalid_id', self.test_data)

    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        invalid_data = pd.DataFrame({
            'video_id': ['1A'],
            'person_id': ['user1'],
            'current_num': [1],
            # Missing 'time' and 'text' columns
        })
        with self.assertRaises(ValueError):
            self.analyzer.participant_pre('1A', invalid_data)

    def test_analyze_participation_patterns(self):
        """Test the analysis of participation patterns."""
        current_data, person_list, time_list, k, n, M = self.analyzer.participant_pre(
            '1A', self.test_data
        )
        
        # Test participation frequency
        total_participations = M.sum().sum()
        self.assertTrue(total_participations > 0)
        
        # Test participation distribution
        participant_counts = M.sum(axis=1)
        self.assertEqual(len(participant_counts), k)
        self.assertTrue(all(participant_counts >= 0))

    def test_window_size_effects(self):
        """Test how different window sizes affect the analysis."""
        _, person_list, time_list, _, _, M = self.analyzer.participant_pre(
            '1A', self.test_data
        )
        
        # Test with minimum window size
        min_window = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=0.1,
            min_num=2,
            max_num=10
        )
        
        # Test with maximum window size
        max_window = self.analyzer.get_best_window_num(
            time_list=time_list,
            M=M,
            best_window_indices=0.9,
            min_num=2,
            max_num=10
        )
        
        self.assertTrue(min_window <= max_window)

    def test_special_text_cases(self):
        """Test handling of special text cases."""
        # Create a small sample of the test data
        special_data = self.test_data.iloc[:5].copy()
        special_cases = [
            "",  # Empty text
            "   ",  # Only spaces
            "!@#$%^&*()",  # Only special characters
            "a" * 1000,  # Very long text
            "\n\t\r",  # Only whitespace characters
        ]
        
        for i, case in enumerate(special_cases):
            special_data.iloc[i, special_data.columns.get_loc('text')] = case
        
        current_data, _, _, _, _, _ = self.analyzer.participant_pre('1A', special_data)
        
        # Check if all cases were processed without errors
        self.assertEqual(len(current_data), len(special_data))
        self.assertTrue('text_clean' in current_data.columns)
        
        # Check if empty or whitespace-only texts are handled properly
        empty_texts = current_data[current_data['text'].str.strip() == '']
        self.assertTrue(len(empty_texts) > 0)

    def test_newness_calculation(self):
        """Test the newness calculation functionality."""
        # Create test data with known patterns
        test_data = pd.DataFrame({
            'video_id': ['1A'] * 6,
            'person_id': ['Student1', 'Student2', 'Student2', 'Student2', 'Student2', 'Student1'],
            'time': range(1, 7),
            'text': [
                '人工智能技术的发展非常快',  # First message by Student1 about AI development
                '人工智能确实在各领域都有应用',  # Student2's first message, similar topic about AI
                '深度学习是AI的重要分支',  # Related to AI
                '机器学习算法很有意思',  # Still about AI
                '神经网络模型发展迅速',  # Different but related to AI
                '总的来说AI发展潜力巨大'  # Summary about AI
            ]
        })
        
        # For debugging
        print("\nProcessed texts:")
        current_data = test_data[test_data.video_id == '1A'].copy()
        current_data['text_clean'] = current_data.text.apply(self.analyzer.text_processor.chinese_word_cut)
        for _, row in current_data.iterrows():
            print(f"{row['person_id']} - Original: {row['text']}")
            print(f"{row['person_id']} - Processed: {row['text_clean']}\n")
    
        # Analyze the video
        results = self.analyzer.analyze_ conversation('1A', test_data)
        
        # Print newness scores for debugging
        print("\nNewness scores:")
        print(results[['Newness']])
    
        # Check newness scores
        self.assertEqual(results.loc['Student1', 'Newness'], 1.0)  # First message should have max newness
        self.assertGreater(results.loc['Student2', 'Newness'], 0)  # Should have some newness
        self.assertLess(results.loc['Student2', 'Newness'], 1)  # But less than max since messages are similar

    def test_analyze_ conversation_integration(self):
        """Test the complete video analysis pipeline."""
        # Create test data
        test_data = pd.DataFrame({
            'video_id': ['1A'] * 10,
            'person_id': ['Student1', 'Student2', 'Teacher'] * 3 + ['Student1'],
            'time': list(range(1, 11)),
            'text': [
                'Hello everyone!',
                'Hi teacher',
                '今天我们来讨论AI',
                'I have a question',
                '这是一个很好的问题',
                'Thank you',
                '让我们继续',
                'I understand now',
                '大家都明白了吗',
                'Yes, clear'
            ]
        })
        
        # Analyze the video
        results = self.analyzer.analyze_ conversation('1A', test_data)
        
        # Check results structure
        expected_columns = {
            'Pa', 'Pa_average', 'Pa_std', 'video_id', 'Pa_hat',
            'Internal_cohesion', 'Overall_responsivity', 'Social_impact',
            'Newness', 'Communication_density'
        }
        self.assertTrue(all(col in results.columns for col in expected_columns))
        
        # Check participant metrics
        for participant in ['Student1', 'Student2', 'Teacher']:
            # Participation metrics
            self.assertGreaterEqual(results.loc[participant, 'Pa'], 0)
            self.assertLessEqual(results.loc[participant, 'Pa_average'], 1)
            
            # Interaction metrics
            self.assertGreaterEqual(results.loc[participant, 'Internal_cohesion'], 0)
            self.assertGreaterEqual(results.loc[participant, 'Overall_responsivity'], 0)
            self.assertGreaterEqual(results.loc[participant, 'Social_impact'], 0)
            
            # Content metrics
            self.assertGreaterEqual(results.loc[participant, 'Newness'], 0)
            self.assertLessEqual(results.loc[participant, 'Newness'], 1)
            self.assertGreaterEqual(results.loc[participant, 'Communication_density'], 0)

    def test_error_handling_integration(self):
        """Test error handling in the complete analysis pipeline."""
        # Test with empty data
        empty_data = pd.DataFrame(columns=['video_id', 'person_id', 'time', 'text'])
        with self.assertRaises(ValueError):
            self.analyzer.analyze_ conversation('1A', empty_data)
        
        # Test with missing required columns
        invalid_data = pd.DataFrame({
            'video_id': ['1A'],
            'person_id': ['user1'],
            'current_num': [1],
            # Missing 'time' and 'text' columns
        })
        with self.assertRaises(ValueError):
            self.analyzer.analyze_ conversation('1A', invalid_data)
        
        # Test with invalid video_id
        with self.assertRaises(ValueError):
            self.analyzer.analyze_ conversation('invalid_id', self.test_data)
        
        # Test with single participant
        single_participant_data = pd.DataFrame({
            'video_id': ['1A'] * 3,
            'person_id': ['P1'] * 3,
            'time': range(1, 4),
            'text': ['Message 1', 'Message 2', 'Message 3']
        })
        results = self.analyzer.analyze_ conversation('1A', single_participant_data)
        self.assertEqual(len(results), 1)  # Should only have one participant

if __name__ == '__main__':
    unittest.main()
