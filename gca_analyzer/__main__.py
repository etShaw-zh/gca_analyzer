import os
import argparse
import pandas as pd
from gca_analyzer import (
    Config, WindowConfig, ModelConfig, VisualizationConfig, LoggerConfig,
    GCAAnalyzer, GCAVisualizer, normalize_metrics
)


def main(args=None):
    """Main function to run GCA analysis from command line."""
    if args is None:
        parser = argparse.ArgumentParser(
            description='GCA (Group Communication Analysis) Analyzer'
        )
        # Data arguments
        parser.add_argument(
            '--data', type=str, required=True,
            help='Path to the CSV file containing interaction data'
        )
        parser.add_argument(
            '--output', type=str, default='gca_results',
            help='Directory to save analysis results (default: gca_results)'
        )

        # Window configuration
        parser.add_argument(
            '--best-window-indices', type=float, default=0.3,
            help='Proportion of best window indices (default: 0.3)'
        )
        parser.add_argument(
            '--min-window-size', type=int, default=2,
            help='Minimum window size (default: 2)'
        )
        parser.add_argument(
            '--max-window-size', type=int, default=10,
            help='Maximum window size (default: 10)'
        )

        # Model configuration
        parser.add_argument(
            '--model-name', type=str,
            default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            help='Name of the model to use (default: sentence-transformers/'
                 'paraphrase-multilingual-MiniLM-L12-v2)'
        )
        parser.add_argument(
            '--model-mirror', type=str,
            default='https://modelscope.cn/models',
            help='Mirror URL for model download '
                 '(default: https://modelscope.cn/models)'
        )

        # Visualization configuration
        parser.add_argument(
            '--default-figsize', type=float, nargs=2, default=[10, 8],
            help='Default figure size (width height) (default: 10 8)'
        )
        parser.add_argument(
            '--heatmap-figsize', type=float, nargs=2, default=[10, 6],
            help='Heatmap figure size (width height) (default: 10 6)'
        )

        # Logger configuration
        parser.add_argument(
            '--log-file', type=str,
            help='Path to log file. If not specified, only console output is used'
        )
        parser.add_argument(
            '--console-level', type=str, default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Logging level for console output (default: INFO)'
        )
        parser.add_argument(
            '--file-level', type=str, default='DEBUG',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Logging level for file output (default: DEBUG)'
        )
        parser.add_argument(
            '--log-rotation', type=str, default='10 MB',
            help='Log file rotation setting (default: 10 MB)'
        )
        parser.add_argument(
            '--log-compression', type=str, default='zip',
            help='Log file compression format (default: zip)'
        )

        args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Input file not found: {args.data}")

    output_parent = os.path.dirname(args.output)
    if output_parent and not os.path.exists(output_parent):
        raise OSError(f"Parent directory does not exist: {output_parent}")

    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory {args.output}: {str(e)}")

    df = pd.read_csv(args.data)

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
    config.logger = LoggerConfig(
        console_level=args.console_level,
        file_level=args.file_level,
        log_file=args.log_file,
        rotation=args.log_rotation,
        compression=args.log_compression
    )

    from .logger import setup_logger
    logger = setup_logger(config)

    analyzer = GCAAnalyzer(config=config)
    visualizer = GCAVisualizer(config=config)

    metrics = [
        'participation', 'responsivity', 'internal_cohesion',
        'social_impact', 'newness', 'comm_density'
    ]

    conversation_ids = df['conversation_id'].unique()

    all_metrics = {}
    for conversation_id in conversation_ids:
        print(f"=== Analyzing Conversation {conversation_id} ===")

        metrics_df = analyzer.analyze_conversation(conversation_id, df)

        metrics_df = metrics_df[metrics]

        all_metrics[conversation_id] = metrics_df

        plot_metrics_distribution = visualizer.plot_metrics_distribution(
            normalize_metrics(metrics_df, metrics, inplace=False),
            metrics=metrics,
            title='Distribution of Normalized Interaction Metrics'
        )
        try:
            plot_metrics_distribution.write_html(
                os.path.join(args.output, f'metrics_distribution_{conversation_id}.html')
            )
        except OSError as e:
            raise OSError(f"Failed to save output files in {args.output}: {str(e)}")

        plot_metrics_radar = visualizer.plot_metrics_radar(
            normalize_metrics(metrics_df, metrics, inplace=False),
            metrics=metrics,
            title='Metrics Radar Chart'
        )
        try:
            plot_metrics_radar.write_html(
                os.path.join(args.output, f'metrics_radar_{conversation_id}.html')
            )
        except OSError as e:
            raise OSError(f"Failed to save output files in {args.output}: {str(e)}")

        try:
            metrics_df.to_csv(os.path.join(args.output, f'metrics_{conversation_id}.csv'))
        except OSError as e:
            raise OSError(f"Failed to save output files in {args.output}: {str(e)}")

    analyzer.calculate_descriptive_statistics(all_metrics, args.output)


if __name__ == '__main__':
    main()
