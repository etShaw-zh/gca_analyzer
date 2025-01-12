import os
import argparse
import pandas as pd
from gca_analyzer import (
    Config, WindowConfig, ModelConfig, VisualizationConfig,
    GCAAnalyzer, GCAVisualizer, normalize_metrics
)

def main():
    """
    Main function to run GCA analysis from command line.
    """
    parser = argparse.ArgumentParser(description='GCA (Group Communication Analysis) Analyzer')
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the CSV file containing interaction data')
    parser.add_argument('--output', type=str, default='gca_results',
                      help='Directory to save analysis results (default: gca_results)')
    
    # Window configuration
    parser.add_argument('--best-window-indices', type=float, default=0.3,
                      help='Proportion of best window indices (default: 0.3)')
    parser.add_argument('--min-window-size', type=int, default=2,
                      help='Minimum window size (default: 2)')
    parser.add_argument('--max-window-size', type=int, default=10,
                      help='Maximum window size (default: 10)')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                      default='iic/nlp_gte_sentence-embedding_chinese-base',
                      help='Name of the model to use (default: iic/nlp_gte_sentence-embedding_chinese-base)')
    parser.add_argument('--model-mirror', type=str,
                      default='https://modelscope.cn/models',
                      help='Mirror URL for model download (default: https://modelscope.cn/models)')
    
    # Visualization configuration
    parser.add_argument('--default-figsize', type=float, nargs=2, default=[10, 8],
                      help='Default figure size (width height) (default: 10 8)')
    parser.add_argument('--heatmap-figsize', type=float, nargs=2, default=[10, 6],
                      help='Heatmap figure size (width height) (default: 10 6)')
    
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.data)
    
    # Create configuration
    config = Config()
    config.window = WindowConfig(
        best_window_indices=args.best_window_indices,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size
    )
    config.model = ModelConfig(
        model_name=args.model_name,
        mirror_url=args.model_mirror
    )
    config.visualization = VisualizationConfig(
        default_figsize=tuple(args.default_figsize),
        heatmap_figsize=tuple(args.heatmap_figsize)
    )

    # Initialize analyzer and visualizer
    analyzer = GCAAnalyzer(config=config)
    visualizer = GCAVisualizer(config=config)

    # Define metrics for analysis
    metrics = ['participation', 'responsivity', 'internal_cohesion', 
              'social_impact', 'newness', 'comm_density']

    # Get list of conversation IDs
    conversation_ids = df['conversation_id'].unique()
    
    # Create results directory
    os.makedirs(args.output, exist_ok=True)

    # Analyze metrics for each conversation
    all_metrics = {}
    for conversation_id in conversation_ids:
        print(f"=== Analyzing Conversation {conversation_id} ===")
        
        # Run GCA analysis
        metrics_df = analyzer.analyze_conversation(conversation_id, df)
        
        # Rename columns to match required metric names
        metrics_df = metrics_df.rename(columns={
            'Pa_hat': 'participation',
            'Overall_responsivity': 'responsivity',
            'Internal_cohesion': 'internal_cohesion',
            'Social_impact': 'social_impact',
            'Newness': 'newness',
            'Communication_density': 'comm_density'
        })
        
        # Keep only the required metrics
        metrics_df = metrics_df[metrics]
        
        all_metrics[conversation_id] = metrics_df

        # Generate and save the distribution plot
        plot_metrics_distribution = visualizer.plot_metrics_distribution(
            normalize_metrics(metrics_df, metrics, inplace=False), 
            metrics=metrics,
            title='Distribution of Normalized Interaction Metrics'
        )
        plot_metrics_distribution.write_html(os.path.join(args.output, f'metrics_distribution_{conversation_id}.html'))

        # Create metrics radar chart
        plot_metrics_radar = visualizer.plot_metrics_radar(
            normalize_metrics(metrics_df, metrics, inplace=False),
            metrics=metrics,
            title='Metrics Radar Chart'
        )
        plot_metrics_radar.write_html(os.path.join(args.output, f'metrics_radar_{conversation_id}.html'))
        
        # Save detailed results to CSV
        metrics_df.to_csv(os.path.join(args.output, f'metrics_{conversation_id}.csv'))
    
    # Merge data from all conversations
    all_data = pd.concat(all_metrics.values())
    
    # Calculate descriptive statistics
    stats = pd.DataFrame({
        'Minimum': all_data.min(),
        'Median': all_data.median(),
        'M': all_data.mean(),
        'SD': all_data.std(),
        'Maximum': all_data.max()
    }).round(2)
    
    measure_order = [
        'participation',
        'social_impact',
        'responsivity',
        'internal_cohesion',
        'newness',
        'comm_density'
    ]
    stats = stats.reindex(measure_order)
    
    # Print descriptive statistics
    print("\n=== Descriptive statistics for GCA measures ===")
    print("Measure".ljust(20), end='')
    print("Minimum  Median  M      SD     Maximum")
    print("-" * 60)
    
    for measure in measure_order:
        row = stats.loc[measure]
        print(f"{measure.replace('_', ' ').title().ljust(20)}"
              f"{row['Minimum']:7.2f}  "
              f"{row['Median']:6.2f}  "
              f"{row['M']:5.2f}  "
              f"{row['SD']:5.2f}  "
              f"{row['Maximum']:7.2f}")
    
    # Save summary results
    stats.to_csv(os.path.join(args.output, 'descriptive_statistics.csv'))

if __name__ == '__main__':
    main()
