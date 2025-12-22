#!/usr/bin/env python3
"""
Check FreeSound URLs directly from HuggingFace dataset.
Streams the dataset without downloading audio, validates URLs efficiently.

Uses columns=["username", "freesound_id"] to avoid loading audio data,
making it much faster and memory-efficient than processing JSON files.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import requests
from tqdm import tqdm

# Configuration
OUTPUT_JSON = Path("freesound_url_validation.json")
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds - base delay between retries
RATE_LIMIT_DELAY = 2  # seconds - minimum delay between requests
COOLDOWN_DURATION = (10, 30)  # seconds - range for cooldown after rate limit
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def construct_freesound_url(username: str, freesound_id: int) -> str:
    """Construct FreeSound URL from username and FreeSound ID."""
    return f"https://freesound.org/people/{username}/sounds/{freesound_id}/"


def test_url_session(
    session: requests.Session, url: str, verbose: bool = False
) -> Tuple[bool, int, str, bool]:
    """
    Test a single URL with retry logic.

    Returns:
        Tuple of (is_valid, status_code, error_message, should_cooldown)
        should_cooldown indicates if we should pause longer before next URL
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Add random jitter to avoid pattern detection
            jitter = random.uniform(0.5, 1.5)
            time.sleep(RATE_LIMIT_DELAY * jitter)

            # Make HEAD request first (lighter)
            response = session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            status_code = response.status_code

            if status_code == 200:
                return True, status_code, "", False
            elif status_code == 404:
                return False, status_code, "Page not found", False
            elif status_code in [403, 429]:
                # Rate limited or blocked
                if verbose:
                    print(
                        f"  Rate limited (attempt {attempt + 1}/{MAX_RETRIES}): {status_code}"
                    )

                # Exponential backoff with jitter
                delay = RETRY_DELAY_BASE * (2**attempt) * random.uniform(0.8, 1.2)
                time.sleep(delay)

                if attempt == MAX_RETRIES - 1:
                    # After max retries, return with cooldown flag
                    return (
                        False,
                        status_code,
                        f"Rate limited after {MAX_RETRIES} attempts",
                        True,  # should_cooldown
                    )
            else:
                # Other errors (500, etc.)
                return False, status_code, f"HTTP {status_code}", False

        except requests.exceptions.Timeout:
            if verbose:
                print(f"  Timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                return False, 0, "Timeout", False
            time.sleep(RETRY_DELAY_BASE * (2**attempt))

        except requests.exceptions.ConnectionError:
            if verbose:
                print(f"  Connection error (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                return False, 0, "Connection error", False
            time.sleep(RETRY_DELAY_BASE * (2**attempt))

        except requests.exceptions.RequestException as e:
            return False, 0, str(e), False

    return False, 0, "Unknown error", False


def check_dataset_urls(
    max_items: int = None,
    max_reads: int = None,
    batch_size: int = 50,
    verbose: bool = False,
    random_sample: int = None,
    output_path: Path = OUTPUT_JSON,
) -> Dict:
    """
    Stream dataset from HuggingFace and check URLs.

    Args:
        max_items: Maximum number of items to process (None = all)
        batch_size: Save checkpoint every N items
        verbose: Print detailed progress
        random_sample: If set, randomly sample this many items from dataset

    Returns:
        Dictionary with validation results
    """
    print("=" * 70)
    print("FreeSound URL Validator - Direct from HuggingFace")
    print("=" * 70)

    # Load dataset (streaming mode - no audio download)
    print("\nLoading dataset from HuggingFace...")
    print("Dataset: benjamin-paine/freesound-laion-640k")
    print("Mode: Streaming (no audio downloaded)")
    print("Columns: username, freesound_id (audio data excluded)")

    # Import here so importing the module doesn't require `datasets` to be installed
    try:
        from datasets import load_dataset
    except Exception as e:  # ImportError or others
        raise RuntimeError(
            "The 'datasets' package is required to stream the HF dataset. "
            "Install it with 'uv sync' or 'pip install datasets'"
        ) from e

    dataset = load_dataset(
        "benjamin-paine/freesound-laion-640k",
        split="train",
        streaming=True,
        columns=["username", "freesound_id"],  # Only load needed columns
    )

    # If random sampling, we need to load all IDs first
    if random_sample:
        print(
            f"\nCollecting dataset IDs for random sampling ({random_sample} items)..."
        )
        all_ids = []
        for item in dataset:
            username = item.get("username", "")
            freesound_id = item.get("freesound_id", 0)
            if username and freesound_id:
                all_ids.append((username, freesound_id))
            if len(all_ids) >= random_sample * 10:  # Get extra for filtering
                break

        if len(all_ids) < random_sample:
            print(f"Warning: Only found {len(all_ids)} items, using all available")
        else:
            print(f"Found {len(all_ids)} items, sampling {random_sample}")

        # Randomly sample
        sampled_ids = random.sample(all_ids, min(random_sample, len(all_ids)))
        test_items = [
            {"username": username, "freesound_id": freesound_id}
            for username, freesound_id in sampled_ids
        ]
    else:
        test_items = dataset

    # Setup session
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )

    # Results tracking
    results = {
        "total_tested": 0,
        "items_seen": 0,
        "valid": 0,
        "not_found": 0,
        "rate_limited": 0,
        "other_errors": 0,
        "details": [],
        "summary_by_status": {},
        "dataset_source": "benjamin-paine/freesound-laion-640k",
    }

    print("\nStarting URL validation...")
    print(f"Rate limiting: {RATE_LIMIT_DELAY}s between requests")
    print(f"Timeout: {REQUEST_TIMEOUT}s per request")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Checkpoint every: {batch_size} items")

    # Progress bar (iterate test_items whether sampled or streaming)
    if random_sample:
        pbar = tqdm(test_items, desc="Testing URLs", unit="url")
    else:
        pbar = tqdm(test_items, desc="Processing dataset", unit="items")

    for i, item in enumerate(pbar):
        # Count how many raw dataset items we've seen
        results["items_seen"] += 1

        # Stop if we've read too many raw items (useful to limit scanning)
        if max_reads and results["items_seen"] >= max_reads:
            if verbose:
                print(f"Reached max reads: {results['items_seen']}/{max_reads}")
            break

        # Check if we've reached max_items (number of URLs tested)
        if max_items and results["total_tested"] >= max_items:
            break

        # Extract metadata
        username = item.get("username", "")
        freesound_id = item.get("freesound_id", 0)

        # Skip invalid entries
        if not username or not freesound_id:
            continue

        # Construct URL
        url = construct_freesound_url(username, freesound_id)

        # Test the URL
        is_valid, status_code, error_msg, should_cooldown = test_url_session(
            session, url, verbose
        )

        # Update results
        results["total_tested"] += 1
        if verbose:
            print(
                f"Tested #{results['total_tested']:d}: {url} -> {status_code} {error_msg}"
            )

        status_category = "valid" if is_valid else "other_errors"
        if status_code == 404:
            status_category = "not_found"
        elif status_code in [403, 429]:
            status_category = "rate_limited"

        results[status_category] += 1

        # Track by status code
        if status_code not in results["summary_by_status"]:
            results["summary_by_status"][status_code] = 0
        results["summary_by_status"][status_code] += 1

        # Store detailed result
        results["details"].append(
            {
                "username": username,
                "freesound_id": freesound_id,
                "url": url,
                "valid": is_valid,
                "status_code": status_code,
                "error": error_msg,
            }
        )

        # Update progress bar
        postfix = {
            "seen": results["items_seen"],
            "valid": results["valid"],
            "404": results["not_found"],
            "limited": results["rate_limited"],
            "errors": results["other_errors"],
        }

        if random_sample:
            pbar.set_postfix(postfix)
        else:
            pbar.set_postfix(postfix)

        # Save checkpoint periodically (avoid saving at 0)
        if results["total_tested"] > 0 and (results["total_tested"] % batch_size == 0):
            save_results(results, output_path)
            print(
                f"\n  Checkpoint saved after {results['total_tested']} URLs -> {output_path}"
            )

        # Handle cooldown if needed
        if should_cooldown:
            cooldown_time = random.uniform(COOLDOWN_DURATION[0], COOLDOWN_DURATION[1])
            if verbose:
                print(f"  Rate limited - cooling down for {cooldown_time:.1f} seconds")
            time.sleep(cooldown_time)

        # Check for max limits after handling cooldown
        if max_items and results["total_tested"] >= max_items:
            print(
                f"Reached max_items: tested {results['total_tested']}/{max_items}. Stopping."
            )
            save_results(results, output_path)
            break
        if max_reads and results["items_seen"] >= max_reads:
            print(
                f"Reached max_reads: seen {results['items_seen']}/{max_reads}. Stopping."
            )
            save_results(results, output_path)
            break

    # Save final results
    save_results(results, output_path)

    return results


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_summary(results: Dict):
    """Print a summary of the validation results."""
    print("\n" + "=" * 70)
    print("URL Validation Summary")
    print("=" * 70)

    total = results["total_tested"]
    if total == 0:
        print("No URLs tested.")
        return

    print(f"Dataset: {results['dataset_source']}")
    print(f"Total URLs tested: {total:,}")
    print(
        f"Valid (200):       {results['valid']:,} ({results['valid'] / total * 100:.1f}%)"
    )
    print(
        f"Not Found (404):   {results['not_found']:,} ({results['not_found'] / total * 100:.1f}%)"
    )
    print(
        f"Rate Limited:      {results['rate_limited']:,} ({results['rate_limited'] / total * 100:.1f}%)"
    )
    print(
        f"Other Errors:      {results['other_errors']:,} ({results['other_errors'] / total * 100:.1f}%)"
    )

    print("\nStatus Code Breakdown:")
    for status_code, count in sorted(results["summary_by_status"].items()):
        print(f"  {status_code}: {count:,}")

    print("\n" + "=" * 70)

    # Show some examples
    if results["not_found"] > 0:
        print("\nExample 404 URLs:")
        for detail in results["details"][:5]:
            if not detail["valid"] and detail["status_code"] == 404:
                print(f"  - {detail['url']}")

    if results["rate_limited"] > 0:
        print("\nRate limited URLs:")
        for detail in results["details"][:5]:
            if detail["status_code"] in [403, 429]:
                print(f"  - {detail['url']} (HTTP {detail['status_code']})")

    print("\n" + "=" * 70)


def filter_valid_urls(results: Dict, output_path: Path):
    """Create a new JSON with only valid URLs."""
    valid_entries = []

    for detail in results["details"]:
        if detail["valid"]:
            valid_entries.append(
                {
                    "username": detail["username"],
                    "freesound_id": detail["freesound_id"],
                    "url": detail["url"],
                    "status": "valid",
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_entries, f, indent=2)

    print(f"\nSaved {len(valid_entries)} valid URLs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check FreeSound URLs directly from HuggingFace dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_JSON),
        help=f"Output results file (default: {OUTPUT_JSON})",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to process (default: all)",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=None,
        help="Maximum number of raw dataset items to read from the HuggingFace stream (default: all)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N items from dataset (useful for quick tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Save checkpoint every N URLs (default: 50)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress for each URL"
    )
    parser.add_argument(
        "--filter-valid", type=str, help="Save valid URLs to a separate JSON file"
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Run validation
    try:
        results = check_dataset_urls(
            max_items=args.max_items,
            max_reads=args.max_reads,
            batch_size=args.batch_size,
            verbose=args.verbose,
            random_sample=args.sample,
            output_path=output_path,
        )
        # Ensure results are saved to requested output path (extra safety)
        save_results(results, output_path)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
        if "results" in locals():
            save_results(results, output_path)
            print_summary(results)
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    # Print summary
    print_summary(results)

    # Filter valid URLs if requested
    if args.filter_valid:
        filter_valid_urls(results, Path(args.filter_valid))

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
