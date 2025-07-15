"""Main CLI implementation for GCA Analyzer.

This module contains the core CLI functionality, including argument parsing,
interactive configuration, validation, and analysis execution.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-07-14
License: Apache 2.0
"""

import argparse
import os
import sys
from io import StringIO
from typing import Optional, Union

import pandas as pd

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from gca_analyzer import (Config, GCAAnalyzer, GCAVisualizer, LoggerConfig,
                          ModelConfig, VisualizationConfig, WindowConfig,
                          normalize_metrics)

# Initialize rich console
console = Console()


def get_sample_data_path() -> str:
    """Get the path to the built-in sample data file."""
    try:
        # Try using importlib.resources first (modern approach)
        package_files = files("gca_analyzer")
        data_file = package_files / "data" / "sample_conversation.csv"
        return str(data_file)
    except Exception as e:
        # Fallback to relative path
        show_error(f"Error reading sample data: {str(e)}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "data", "sample_conversation.csv")


def show_sample_data_preview():
    """Display a preview of the sample data."""
    sample_path = get_sample_data_path()

    if not os.path.exists(sample_path):
        show_error("Sample data file not found")
        return

    try:
        df = pd.read_csv(sample_path)

        # Verify required columns exist
        required_columns = {"conversation_id", "person_id", "text", "time"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in the sample data: "
                f"{', '.join(missing_columns)}"
            )
        # Show sample data info
        console.print(
            Panel(
                f"[bold cyan]📊 Sample Data Preview[/bold cyan]\n\n"
                f"[green]✅ File:[/green] {sample_path}\n"
                f"[green]✅ Records:[/green] {len(df)}\n"
                f"[green]✅ Conversations:[/green] "
                f"{len(df['conversation_id'].unique())}\n"
                f"[green]✅ Participants:[/green] "
                f"{len(df['person_id'].unique())}\n\n"
                f"[dim]Conversation Types:[/dim] "
                f"{', '.join(df['conversation_id'].unique())}",
                title="Sample Data",
                border_style="cyan",
            )
        )

        # Show first few rows
        console.print("\n[bold]First 5 rows:[/bold]")
        preview_table = Table(show_header=True, border_style="blue")
        preview_table.add_column("Conversation ID", style="cyan")
        preview_table.add_column("Person ID", style="yellow")
        preview_table.add_column("Text", style="white", max_width=50)
        preview_table.add_column("Time", style="green")

        for _, row in df.head(5).iterrows():
            preview_table.add_row(
                row["conversation_id"],
                row["person_id"],
                row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"],
                row["time"],
            )

        console.print(preview_table)

    except Exception as e:
        show_error(f"Error reading sample data: {str(e)}")


def show_welcome():
    """Display a beautiful welcome message."""
    welcome_text = Text()
    welcome_text.append("GCA Analyzer", style="bold blue")
    welcome_text.append(" - Group Communication Analysis Tool", style="dim")

    welcome_panel = Panel(
        welcome_text, title="🔍 Welcome", border_style="blue", padding=(1, 2)
    )
    console.print(welcome_panel)
    console.print()


def show_error(message: str):
    """Display error message with rich formatting."""
    error_panel = Panel(
        f"[red]❌ Error:[/red] {message}", border_style="red", title="Error"
    )
    console.print(error_panel)


def show_success(message: str):
    """Display success message with rich formatting."""
    success_panel = Panel(
        f"[green]✅ Success:[/green] {message}", border_style="green", title="Success"
    )
    console.print(success_panel)


def show_info(message: str):
    """Display info message with rich formatting."""
    console.print(f"[cyan]ℹ️  {message}[/cyan]")


def interactive_config_wizard() -> Optional[argparse.Namespace]:
    """Interactive configuration wizard for new users."""
    console.print("\n[bold cyan]🧙 Interactive Configuration Wizard[/bold cyan]")
    console.print("Let's set up your GCA analysis step by step!\n")

    # Ask if user wants to use sample data
    use_sample = Confirm.ask(
        "[bold]🎯 Would you like to use the built-in sample data?[/bold]", default=True
    )

    if use_sample:
        sample_path = get_sample_data_path()
        if not os.path.exists(sample_path):
            show_error("Sample data file not found")
            return None

        # Show sample data preview
        show_sample_data_preview()

        # Ask for confirmation
        if not Confirm.ask(
            "\n[bold]Continue with this sample data?[/bold]", default=True
        ):
            return None

        data_path = sample_path
    else:
        # Data file path
        data_path = Prompt.ask(
            "[bold]📁 Enter the path to your CSV data file[/bold]",
            default="example/data/test_data.csv",
        )

        if not os.path.exists(data_path):
            show_error(f"Input file not found: {data_path}")
            return None

    # Output directory
    output_dir = Prompt.ask(
        "[bold]📂 Enter the output directory[/bold]", default="gca_results"
    )

    # Advanced configuration
    use_advanced = Confirm.ask(
        "[bold]⚙️  Configure advanced settings?[/bold]", default=False
    )

    # Create args namespace
    args = argparse.Namespace()
    args.data = data_path
    args.output = output_dir
    args.interactive = True
    args.sample_data = use_sample

    if use_advanced:
        # Window configuration
        console.print("\n[bold yellow]🪟 Window Configuration[/bold yellow]")
        args.best_window_indices = float(
            Prompt.ask("Best window indices proportion", default="0.3")
        )
        args.act_participant_indices = int(
            Prompt.ask("Active participant indices", default="2")
        )
        args.min_window_size = int(Prompt.ask("Minimum window size", default="2"))
        max_window = Prompt.ask(
            "Maximum window size (press Enter for auto)", default=""
        )
        args.max_window_size = int(max_window) if max_window else None

        # Model configuration
        console.print("\n[bold yellow]🤖 Model Configuration[/bold yellow]")
        args.model_name = Prompt.ask(
            "Model name",
            default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        args.model_mirror = Prompt.ask(
            "Model mirror URL", default="https://modelscope.cn/models"
        )

        # Logging configuration
        console.print("\n[bold yellow]📝 Logging Configuration[/bold yellow]")
        args.console_level = Prompt.ask(
            "Console log level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
        )
        args.log_file = (
            Prompt.ask("Log file path (press Enter to skip)", default="") or None
        )
        args.file_level = "DEBUG"
        args.log_rotation = "10 MB"
        args.log_compression = "zip"
    else:
        # Use defaults
        args.best_window_indices = 0.3
        args.act_participant_indices = 2
        args.min_window_size = 2
        args.max_window_size = None
        args.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        args.model_mirror = "https://modelscope.cn/models"
        args.default_figsize = [10, 8]
        args.heatmap_figsize = [10, 6]
        args.console_level = "INFO"
        args.log_file = None
        args.file_level = "DEBUG"
        args.log_rotation = "10 MB"
        args.log_compression = "zip"

    # Set visualization defaults if not set
    if not hasattr(args, "default_figsize"):
        args.default_figsize = [10, 8]
    if not hasattr(args, "heatmap_figsize"):
        args.heatmap_figsize = [10, 6]

    return args


def validate_inputs(args) -> bool:
    """Validate input arguments with rich error reporting."""
    errors = []

    if args.data and not os.path.exists(args.data):
        errors.append(f"Input file not found: {args.data}")

    output_parent = os.path.dirname(args.output)
    if output_parent and not os.path.exists(output_parent):
        errors.append(f"Parent directory does not exist: {output_parent}")

    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as e:  # pragma: no cover
        errors.append(f"Failed to create output directory {args.output}: {str(e)}")

    if errors:
        for error in errors:
            show_error(error)
        return False

    return True


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="GCA (Group Communication Analysis) Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gca_analyzer --sample-data --output results/  # Use built-in sample data
  python -m gca_analyzer --data data.csv --output results/
  python -m gca_analyzer --interactive  # Interactive mode
  python -m gca_analyzer -i            # Interactive mode (short)
  python -m gca_analyzer --sample-data --preview  # Preview sample data
                """,
    )

    # Interactive mode arguments
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive configuration mode",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the CSV file containing interaction data",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use built-in sample data instead of providing a data file",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of sample data and exit (use with --sample-data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gca_results",
        help="Directory to save analysis results (default: gca_results)",
    )

    # Window configuration
    parser.add_argument(
        "--best-window-indices",
        type=float,
        default=0.3,
        help="Proportion of best window indices (default: 0.3)",
    )
    parser.add_argument(
        "--act-participant-indices",
        type=int,
        default=2,
        help="Number of contributions from each participant in a window that "
        "is greater than or equal to the active participants threshold "
        "(e.g., at least two contributions). Defaults to 2.",
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=2,
        help="Minimum window size (default: 2)",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=None,
        help="Maximum window size (default: None | if None, max_window_size = len(data))",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Name of the model to use "
        "(default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--model-mirror",
        type=str,
        default="https://modelscope.cn/models",
        help="Mirror URL for model download " "(default: https://modelscope.cn/models)",
    )

    # Visualization configuration
    parser.add_argument(
        "--default-figsize",
        type=float,
        nargs=2,
        default=[10, 8],
        help="Default figure size (width height) (default: 10 8)",
    )
    parser.add_argument(
        "--heatmap-figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        help="Heatmap figure size (width height) (default: 10 6)",
    )

    # Logger configuration
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file. If not specified, only console output is used",
    )
    parser.add_argument(
        "--console-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for console output (default: INFO)",
    )
    parser.add_argument(
        "--file-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for file output (default: DEBUG)",
    )
    parser.add_argument(
        "--log-rotation",
        type=str,
        default="10 MB",
        help="Log file rotation setting (default: 10 MB)",
    )
    parser.add_argument(
        "--log-compression",
        type=str,
        default="zip",
        help="Log file compression format (default: zip)",
    )

    return parser


def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        show_info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        show_success(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:  # pragma: no cover
        show_error(f"Failed to load data: {str(e)}")
        return None


def initialize_components(
    config: Config,
) -> Union[tuple[GCAAnalyzer, GCAVisualizer], tuple[None, None]]:
    """Initialize analyzer and visualizer components."""
    try:
        # Suppress stdout temporarily during model loading
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        analyzer = GCAAnalyzer(config=config)
        visualizer = GCAVisualizer(config=config)

        # Restore stdout
        sys.stdout = old_stdout
        show_success("Analyzer and visualizer initialized successfully!")

        return analyzer, visualizer

    except Exception as e:  # pragma: no cover
        # Restore stdout in case of error
        sys.stdout = old_stdout
        show_error(f"Failed to initialize analyzer: {str(e)}")
        return None, None


def show_configuration_summary(args):
    """Display configuration summary table."""
    config_table = Table(title="🔧 Configuration Summary", border_style="blue")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Data File", args.data)
    config_table.add_row("Output Directory", args.output)
    config_table.add_row("Model", args.model_name)
    config_table.add_row(
        "Window Size", f"{args.min_window_size} - {args.max_window_size or 'auto'}"
    )
    config_table.add_row("Log Level", args.console_level)

    console.print(config_table)
    console.print()


def create_config(args) -> Config:
    """Create configuration object from arguments."""
    config = Config()
    config.window = WindowConfig(
        best_window_indices=args.best_window_indices,
        act_participant_indices=args.act_participant_indices,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
    )
    config.model = ModelConfig(model_name=args.model_name, mirror_url=args.model_mirror)
    config.visualization = VisualizationConfig(
        default_figsize=tuple(args.default_figsize),
        heatmap_figsize=tuple(args.heatmap_figsize),
    )
    config.logger = LoggerConfig(
        console_level=args.console_level,
        file_level=args.file_level,
        log_file=args.log_file,
        rotation=args.log_rotation,
        compression=args.log_compression,
    )
    return config


def analyze_conversations(analyzer, visualizer, df, args):
    """Analyze all conversations and generate results."""
    metrics = [
        "participation",
        "responsivity",
        "internal_cohesion",
        "social_impact",
        "newness",
        "comm_density",
    ]

    conversation_ids = df["conversation_id"].unique()
    show_info(f"Found {len(conversation_ids)} conversations to analyze")

    all_metrics = {}

    # Create progress bar for conversation analysis
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:

        analysis_task = progress.add_task(
            "[cyan]Analyzing conversations...", total=len(conversation_ids)
        )

        for conversation_id in conversation_ids:
            progress.update(
                analysis_task,
                description=f"[cyan]Analyzing conversation {conversation_id}...",
            )

            try:
                # Suppress stdout during analysis to avoid mixed output
                old_stdout = sys.stdout
                sys.stdout = StringIO()

                # Analyze conversation
                metrics_df = analyzer.analyze_conversation(conversation_id, df)
                metrics_df = metrics_df[metrics]
                all_metrics[conversation_id] = metrics_df

                # Generate visualizations
                plot_metrics_distribution = visualizer.plot_metrics_distribution(
                    normalize_metrics(metrics_df, metrics, inplace=False),
                    metrics=metrics,
                    title="Distribution of Normalized Interaction Metrics",
                )

                plot_metrics_radar = visualizer.plot_metrics_radar(
                    normalize_metrics(metrics_df, metrics, inplace=False),
                    metrics=metrics,
                    title="Metrics Radar Chart",
                )

                # Restore stdout
                sys.stdout = old_stdout

                # Save files
                plot_metrics_distribution.write_html(
                    os.path.join(
                        args.output, f"metrics_distribution_{conversation_id}.html"
                    )
                )
                plot_metrics_radar.write_html(
                    os.path.join(args.output, f"metrics_radar_{conversation_id}.html")
                )
                metrics_df.to_csv(
                    os.path.join(args.output, f"metrics_{conversation_id}.csv")
                )

            except Exception as e:
                # Restore stdout in case of error
                sys.stdout = old_stdout
                progress.stop()
                show_error(f"Error processing conversation {conversation_id}: {str(e)}")
                progress.start()
                continue

            progress.advance(analysis_task)

    return all_metrics


def generate_statistics(analyzer, all_metrics, output_dir):
    """Generate descriptive statistics."""
    show_info("Generating descriptive statistics...")
    try:
        # Suppress stdout during statistics generation
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        analyzer.calculate_descriptive_statistics(all_metrics, output_dir)

        # Restore stdout
        sys.stdout = old_stdout

        show_success(f"Analysis completed! Results saved to {output_dir}")

        # Show summary table
        summary_table = Table(title="📊 Analysis Summary", border_style="green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Conversations Analyzed", str(len(all_metrics)))
        summary_table.add_row("Output Directory", output_dir)
        summary_table.add_row("Files Generated", f"{len(all_metrics) * 3 + 1} files")

        console.print(summary_table)

        # Show final completion message
        console.print("\n[bold green]🎉 GCA Analysis Complete![/bold green]")
        console.print(
            f"[dim]Check the '{output_dir}' directory for all generated files.[/dim]"
        )

    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        show_error(f"Error generating statistics: {str(e)}")


def main_cli(args=None):
    """Main CLI function."""
    # Show welcome message
    show_welcome()

    if args is None:  # pragma: no cover
        parser = create_argument_parser()
        args = parser.parse_args()

        # Handle sample data preview
        if args.sample_data and args.preview:
            show_sample_data_preview()
            return

        # Handle sample data usage
        if args.sample_data:
            sample_path = get_sample_data_path()
            if not os.path.exists(sample_path):
                show_error("Sample data file not found")
                return
            args.data = sample_path
            show_info(f"Using built-in sample data: {sample_path}")

        # Check if user wants interactive mode
        if len(sys.argv) == 1 or args.interactive:
            args = interactive_config_wizard()
            if args is None:
                console.print("[red]❌ Configuration cancelled.[/red]")
                return

        # Validate required arguments for non-interactive mode
        if (
            not getattr(args, "interactive", False)
            and not args.data
            and not args.sample_data
        ):
            show_error(
                "--data argument is required in non-interactive mode (or use --sample-data)"
            )
            return

    # Validate inputs using rich formatting
    if not validate_inputs(args):
        return

    # Load data
    df = load_data(args.data)
    if df is None:
        return  # pragma: no cover

    # Show configuration summary
    show_configuration_summary(args)

    # Create configuration
    show_info("Setting up configuration...")
    config = create_config(args)

    from .logger import setup_logger

    setup_logger(config)

    # Initialize components
    show_info("Initializing analyzer and visualizer...")
    analyzer, visualizer = initialize_components(config)
    if analyzer is None or visualizer is None:
        return  # pragma: no cover

    # Analyze conversations
    all_metrics = analyze_conversations(analyzer, visualizer, df, args)

    # Generate statistics
    generate_statistics(analyzer, all_metrics, args.output)
