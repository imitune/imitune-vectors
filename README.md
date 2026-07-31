# ThatSoundsLikeMe vectors

Small scripts for building and uploading audio embedding subsets for ThatSoundsLikeMe.

## Production indexes

- `lenient` (`imitune-search`, **185,803**): Freesound user-tag filtering with the 91-term `config/tags_v1.txt` denylist.
- `strict` (`imitune-search-v2`, **89,905**): older, more aggressive Freesound user-tag filtering with 185 deny terms and partial-string matching; every strict sound is also in lenient.
- `tagged` (`imitune-search-v3`, **187,665**): AST audio-content filtering at threshold 0.30 with the 55 exact labels in `config/filter_labels_exact_v1.txt`.
- `fsd50k` (`imitune-fsd50k`, **51,197**): the separate FSD50K corpus.

Exact overlap by unique Freesound sound ID:

| Sets | Intersection | First only | Second only | Jaccard |
| --- | ---: | ---: | ---: | ---: |
| lenient / tagged | 110,421 | 75,382 | 77,244 | 41.98% |
| lenient / strict | 89,905 | 95,898 | 0 | 48.39% |

The saved 0.25-threshold broad AST experiment is not the deployed `strict` index.

## Layout

- `process_freesound.py`: Hugging Face Freesound tagging, filtering, embedding, upload
- `process_fsd50k.py`: FSD50K embedding pipeline
- `analyze_tag_results.py`: threshold sweeps and review report generation
- `config/`: label lists and filter configs
- `scripts/`: Slurm/job helper scripts
- `docs/`: short workflow notes
- `outputs/`: generated JSON and JSONL artifacts
- `threshold_review.html`: optional top-level review page for threshold tuning

## Common commands

```bash
# Tag once
uv run python process_freesound.py --tag-only --blocked-labels-file config/blocked_model_labels_v1.txt

# Review thresholds
uv run python analyze_tag_results.py \
  --tag-results-jsonl outputs/freesound_tag_results.jsonl \
  --filter-labels-file config/filter_labels_exact_v1.txt \
  --thresholds 0.2 0.25 0.3 \
  --html-report threshold_review.html

# Build embeddings from saved tags
uv run python process_freesound.py \
  --process-only \
  --tag-results-jsonl outputs/freesound_tag_results.jsonl \
  --filter-labels-file config/filter_labels_exact_v1.txt \
  --tagger-threshold 0.30

# Upload existing embeddings
uv run python process_freesound.py --upload-only
```

## Notes

- Generated JSON and JSONL outputs go under `outputs/` and are ignored in git.
- `threshold_review.html` is kept at the repo root as a convenient review artifact.
- FSD50K details live in `docs/FSD50K_WORKFLOW.md`.
