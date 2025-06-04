#!/usr/bin/env python3
"""
Unified Analysis CLI for RED_CORE
Coordinates all analysis tools through a single interface.

Usage:
    python app/analysis/analysis_cli.py quick-insights
    python app/analysis/analysis_cli.py extract-metadata --log-dir experiments/refusal_robustness/logs/
    python app/analysis/analysis_cli.py roll-up --input data/flattened/flat_logs.csv
    python app/analysis/analysis_cli.py review --log-dir experiments/refusal_robustness/logs/
    python app/analysis/analysis_cli.py full-analysis --experiment refusal_robustness
"""

import argparse
import subprocess
import sys
from pathlib import Path
from app.core.logger import get_analysis_logger

logger = get_analysis_logger()


def run_quick_insights():
    """Run the quick log insights report."""
    logger.info("Running quick insights analysis...")
    try:
        subprocess.run([
            sys.executable, 
            "app/analysis/log_insight_report.py"
        ], check=True, cwd=Path.cwd())
    except subprocess.CalledProcessError as e:
        logger.error(f"Quick insights failed: {e}")
        return False
    return True


def extract_metadata(log_dir: str, mode: str = "extract"):
    """Extract metadata using analyze.py."""
    logger.info(f"Extracting metadata from {log_dir}...")
    try:
        subprocess.run([
            sys.executable,
            "app/analysis/analyze.py", 
            "--mode", mode,
            "--log-dir", log_dir
        ], check=True, cwd=Path.cwd())
    except subprocess.CalledProcessError as e:
        logger.error(f"Metadata extraction failed: {e}")
        return False
    return True


def roll_up_data(input_file: str, output_file: str = None):
    """Roll up flattened data using roller.py."""
    logger.info(f"Rolling up data from {input_file}...")
    
    cmd = [
        sys.executable,
        "app/analysis/roller.py",
        "--mode", "collapse", 
        "--input", input_file
    ]
    
    if output_file:
        cmd.extend(["--output", output_file])
    
    try:
        subprocess.run(cmd, check=True, cwd=Path.cwd())
    except subprocess.CalledProcessError as e:
        logger.error(f"Data rollup failed: {e}")
        return False
    return True


def run_enhanced_review(log_dir: str):
    """Run the enhanced review tool."""
    logger.info(f"Starting enhanced review for {log_dir}...")
    try:
        subprocess.run([
            sys.executable,
            "app/analysis/enhanced_review_tool.py",
            "--log-dir", log_dir
        ], check=True, cwd=Path.cwd())
    except subprocess.CalledProcessError as e:
        logger.error(f"Enhanced review failed: {e}")
        return False
    return True


def full_analysis_pipeline(experiment_name: str):
    """Run complete analysis pipeline for an experiment."""
    logger.info(f"Running full analysis pipeline for experiment: {experiment_name}")
    
    log_dir = f"experiments/{experiment_name}/logs/"
    flat_file = f"data/flattened/{experiment_name}_flat_logs.csv" 
    rolled_file = f"data/rolled/{experiment_name}_rolled_logs.csv"
    
    # Ensure directories exist
    Path("data/flattened").mkdir(exist_ok=True)
    Path("data/rolled").mkdir(exist_ok=True)
    
    # Step 1: Extract metadata
    if not extract_metadata(log_dir):
        return False
    
    # Step 2: Roll up data (if flat file exists)
    if Path(flat_file).exists():
        if not roll_up_data(flat_file, rolled_file):
            logger.warning("Data rollup failed, continuing without it")
    else:
        logger.warning(f"Flat file {flat_file} not found, skipping rollup")
    
    # Step 3: Quick insights
    if not run_quick_insights():
        logger.warning("Quick insights failed, continuing")
    
    # Step 4: Enhanced review
    if not run_enhanced_review(log_dir):
        logger.warning("Enhanced review failed")
        return False
    
    logger.info(f"Full analysis pipeline completed for {experiment_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified Analysis CLI for RED_CORE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s quick-insights
  %(prog)s extract-metadata --log-dir experiments/refusal_robustness/logs/
  %(prog)s roll-up --input data/flattened/flat_logs.csv
  %(prog)s review --log-dir experiments/refusal_robustness/logs/  
  %(prog)s full-analysis --experiment refusal_robustness
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis commands')
    
    # Quick insights
    subparsers.add_parser('quick-insights', help='Run quick log insights report')
    
    # Extract metadata
    extract_parser = subparsers.add_parser('extract-metadata', help='Extract log metadata')
    extract_parser.add_argument('--log-dir', required=True, help='Directory containing log files')
    extract_parser.add_argument('--mode', default='extract', help='Analysis mode')
    
    # Roll up data
    rollup_parser = subparsers.add_parser('roll-up', help='Roll up flattened data')
    rollup_parser.add_argument('--input', required=True, help='Input flat CSV file')
    rollup_parser.add_argument('--output', help='Output rolled CSV file')
    
    # Enhanced review
    review_parser = subparsers.add_parser('review', help='Run enhanced review tool')
    review_parser.add_argument('--log-dir', required=True, help='Directory containing log files')
    
    # Full analysis
    full_parser = subparsers.add_parser('full-analysis', help='Run complete analysis pipeline')
    full_parser.add_argument('--experiment', required=True, help='Experiment name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    success = True
    
    if args.command == 'quick-insights':
        success = run_quick_insights()
    elif args.command == 'extract-metadata':
        success = extract_metadata(args.log_dir, args.mode)
    elif args.command == 'roll-up':
        success = roll_up_data(args.input, args.output)
    elif args.command == 'review':
        success = run_enhanced_review(args.log_dir)
    elif args.command == 'full-analysis':
        success = full_analysis_pipeline(args.experiment)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()