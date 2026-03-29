#!/usr/bin/env python3
"""
Download Hindi conversational speech dataset from public GCP mirror.

This script:
1. Reads FT_Data.xlsx
2. Converts restricted GCP URLs to public mirror URLs
3. Downloads all transcription JSON and audio WAV files in parallel
4. Implements resume capability (skips existing files)
5. Logs failed downloads for debugging
"""

import os
import sys
import re
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import traceback

import pandas as pd
import requests
from tqdm import tqdm

# ============================================================================
# Setup
# ============================================================================

def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
EXCEL_FILE = REPO_ROOT / "FT_Data.xlsx"
ERROR_LOG = DATA_DIR / "download_errors.log"

# Create directories
TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(ERROR_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# URL Conversion
# ============================================================================

def to_public_url(gcp_url: str) -> str:
    """
    Convert restricted GCP URL to public mirror URL.
    
    Args:
        gcp_url: URL from joshtalks-data-collection bucket
        
    Returns:
        Corresponding URL from public upload_goai bucket
    """
    return re.sub(
        r'https://storage\.googleapis\.com/joshtalks-data-collection/hq_data/hi/',
        'https://storage.googleapis.com/upload_goai/',
        gcp_url
    )


# ============================================================================
# Download Logic
# ============================================================================

def download_file(
    url: str,
    output_path: Path,
    max_retries: int = 3,
    timeout: int = 30
) -> Tuple[bool, Optional[str]]:
    """
    Download a file with retry logic.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        max_retries: Number of retries on failure
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    # Skip if file already exists and is non-zero
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, None
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Write to temporary file, then rename atomically
            temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
            with open(temp_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Atomic rename
            temp_path.rename(output_path)
            return True, None
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {str(e)}")
                continue
            else:
                return False, str(e)
        except Exception as e:
            return False, str(e)
    
    return False, "Max retries exceeded"


def download_dataset(num_workers: int = 8) -> Tuple[int, int]:
    """
    Download all transcriptions and audio files.
    
    Args:
        num_workers: Number of parallel download threads
        
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    # Read Excel file
    logger.info(f"Reading {EXCEL_FILE}...")
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")
    
    df = pd.read_excel(EXCEL_FILE)
    logger.info(f"Loaded {len(df)} recordings from Excel")
    
    # Prepare download tasks
    tasks = []
    
    for idx, row in df.iterrows():
        recording_id = row['recording_id']
        
        # Transcription JSON
        trans_gcp_url = row['transcription_url_gcp']
        trans_public_url = to_public_url(trans_gcp_url)
        trans_path = TRANSCRIPTIONS_DIR / f"{recording_id}_transcription.json"
        tasks.append((trans_public_url, trans_path, f"transcription_{recording_id}"))
        
        # Audio WAV
        audio_gcp_url = row['rec_url_gcp']
        audio_public_url = to_public_url(audio_gcp_url)
        audio_path = RAW_AUDIO_DIR / f"{recording_id}_audio.wav"
        tasks.append((audio_public_url, audio_path, f"audio_{recording_id}"))
    
    logger.info(f"Total tasks: {len(tasks)} (transcriptions + audio)")
    
    # Execute downloads in parallel
    successful = 0
    failed = 0
    failed_items = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_file, url, path): task_name
            for url, path, task_name in tasks
        }
        
        progress = tqdm(as_completed(futures), total=len(futures), desc="Downloading")
        
        for future in progress:
            task_name = futures[future]
            try:
                success, error_msg = future.result()
                if success:
                    successful += 1
                    progress.update()
                else:
                    failed += 1
                    failed_items.append((task_name, error_msg))
                    logger.error(f"Failed: {task_name} - {error_msg}")
            except Exception as e:
                failed += 1
                failed_items.append((task_name, str(e)))
                logger.error(f"Exception in {task_name}: {traceback.format_exc()}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Successful: {successful}/{len(tasks)}")
    logger.info(f"Failed: {failed}/{len(tasks)}")
    
    if failed_items:
        logger.warning(f"\nFailed items ({len(failed_items)}):")
        for task_name, error in failed_items[:10]:  # Show first 10
            logger.warning(f"  - {task_name}: {error}")
        if len(failed_items) > 10:
            logger.warning(f"  ... and {len(failed_items) - 10} more (see {ERROR_LOG})")
    
    logger.info("=" * 70)
    
    return successful, failed


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Hindi conversational speech dataset"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    try:
        successful, failed = download_dataset(num_workers=args.workers)
        
        # Exit with error code if any downloads failed
        if failed > 0:
            sys.exit(1)
        else:
            logger.info("All downloads completed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
