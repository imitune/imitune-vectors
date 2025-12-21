#!/usr/bin/env python3
"""
Test URLs from the FreeSound embeddings JSON file.
Checks if URLs are valid or return 404 errors.
Includes rate limiting and retry logic to avoid being blocked by FreeSound.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

# Configuration
OUTPUT_JSON = Path("freesound_embeddings.json")
RESULTS_JSON = Path("url_validation_results.json")
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds - base delay between retries
RATE_LIMIT_DELAY = 2  # seconds - minimum delay between requests
COOLDOWN_DURATION = (10, 30)  # seconds - range for cooldown after rate limit
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def load_embeddings(json_path: Path) -> List[Dict]:
    """Load embeddings data from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"Loading embeddings from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} entries")
    return data


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


def test_urls(
    embeddings_data: List[Dict],
    batch_size: int = 50,
    verbose: bool = False,
    random_sample: int = None,
) -> Dict:
    """
    Test all URLs in the embeddings data.

    Args:
        embeddings_data: List of embedding entries
        batch_size: Number of URLs to test before saving checkpoint
        verbose: Print detailed progress
        random_sample: If set, test only this many random entries

    Returns:
        Dictionary with validation results
    """
    if random_sample and random_sample < len(embeddings_data):
        print(
            f"Randomly sampling {random_sample} entries from {len(embeddings_data):,}..."
        )
        import random as py_random

        test_data = py_random.sample(embeddings_data, random_sample)
    else:
        test_data = embeddings_data

    total = len(test_data)
    print(f"\nTesting {total:,} URLs...")
    print(f"Rate limiting: {RATE_LIMIT_DELAY}s between requests")
    print(f"Timeout: {REQUEST_TIMEOUT}s per request")
    print(f"Max retries: {MAX_RETRIES}")

    # Create session with custom headers
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

    results = {
        "total_tested": 0,
        "valid": 0,
        "not_found": 0,
        "rate_limited": 0,
        "other_errors": 0,
        "details": [],
        "summary_by_status": {},
    }

    # Progress bar
    pbar = tqdm(test_data, desc="Testing URLs", unit="url")

    for i, entry in enumerate(pbar):
        url = entry.get("freesound_url", "")
        if not url:
            continue

        # Test the URL
        is_valid, status_code, error_msg, should_cooldown = test_url_session(
            session, url, verbose
        )

        # Update results
        results["total_tested"] += 1

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
                "id": entry.get("id"),
                "url": url,
                "valid": is_valid,
                "status_code": status_code,
                "error": error_msg,
            }
        )

        # Update progress bar description
        pbar.set_postfix(
            {
                "valid": results["valid"],
                "404": results["not_found"],
                "limited": results["rate_limited"],
                "errors": results["other_errors"],
            }
        )

        # Save checkpoint periodically
        if (i + 1) % batch_size == 0:
            save_results(results, RESULTS_JSON)
            if verbose:
                print(f"\n  Checkpoint saved after {i + 1} URLs")

        # Handle cooldown between requests if needed
        if should_cooldown:
            cooldown_time = random.uniform(COOLDOWN_DURATION[0], COOLDOWN_DURATION[1])
            if verbose:
                print(f"  Rate limited - cooling down for {cooldown_time:.1f} seconds")
            time.sleep(cooldown_time)

    # Save final results
    save_results(results, RESULTS_JSON)

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

    # Show some examples of errors
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
            # Find original entry to get full data
            # This is a bit inefficient but keeps things simple
            valid_entries.append(
                {"id": detail["id"], "url": detail["url"], "status": "valid"}
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_entries, f, indent=2)

    print(f"\nSaved {len(valid_entries)} valid URLs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test FreeSound URLs from embeddings JSON file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT_JSON),
        help=f"Input JSON file (default: {OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_JSON),
        help=f"Output results file (default: {RESULTS_JSON})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Test only N random entries (useful for quick tests)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file (skip already tested URLs)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load embeddings
    try:
        embeddings_data = load_embeddings(input_path)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return 1

    # Handle resume
    if args.resume and output_path.exists():
        print(f"Resuming from {output_path}...")
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

        # Get IDs that have already been tested
        tested_ids = {d["id"] for d in existing_results["details"]}
        print(f"Already tested: {len(tested_ids)} URLs")

        # Filter out already tested entries
        embeddings_data = [e for e in embeddings_data if e.get("id") not in tested_ids]
        print(f"Remaining to test: {len(embeddings_data)} URLs")

        # Continue with remaining data
        if len(embeddings_data) == 0:
            print("All URLs already tested. Loading existing results...")
            print_summary(existing_results)
            return 0

        # Run tests on remaining data
        new_results = test_urls(
            embeddings_data,
            batch_size=args.batch_size,
            verbose=args.verbose,
            random_sample=args.sample,
        )

        # Merge results
        merged_results = {
            "total_tested": existing_results["total_tested"]
            + new_results["total_tested"],
            "valid": existing_results["valid"] + new_results["valid"],
            "not_found": existing_results["not_found"] + new_results["not_found"],
            "rate_limited": existing_results["rate_limited"]
            + new_results["rate_limited"],
            "other_errors": existing_results["other_errors"]
            + new_results["other_errors"],
            "details": existing_results["details"] + new_results["details"],
            "summary_by_status": {},
        }

        # Merge status counts
        all_status_codes = set(existing_results["summary_by_status"].keys()) | set(
            new_results["summary_by_status"].keys()
        )
        for code in all_status_codes:
            merged_results["summary_by_status"][code] = existing_results[
                "summary_by_status"
            ].get(code, 0) + new_results["summary_by_status"].get(code, 0)

        results = merged_results
    else:
        # Run fresh tests
        results = test_urls(
            embeddings_data,
            batch_size=args.batch_size,
            verbose=args.verbose,
            random_sample=args.sample,
        )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, output_path)

    # Filter valid URLs if requested
    if args.filter_valid:
        filter_valid_urls(results, Path(args.filter_valid))

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
