---
name: tune-clustering
description: Help the user improve how photos are grouped into people. Use when they report one person split across several folders, different people merged into one, too many photos in unknown/, missing faces in no_faces/, or paste a run summary and ask whether it looks right.
---

# Tuning the grouping

Start from the user's `=== Summary ===` output and what they see in the folders. Ask them for it if they haven't shared it.

## Re-group without re-running detection
Everything below the detection step reads the cache (`faces.npz` / `faces.json`), so a re-run takes seconds:
```bash
docker compose run --rm face-sorter --recluster <flags>
```
Only detection flags need a full run: `--det-thresh`, `--min-face`, `--max-side`, `--det-size`.

## Symptom → change
| Symptom | Try first | Then |
|---|---|---|
| One person in several folders | `--merge-thresh 0.45` (default 0.5) | `--min-samples 1` |
| Two people in one folder (siblings, parent/child) | `--merge-thresh 0.6` (`1` turns merging off) | raise `--min-samples` |
| Many photos in `unknown/` | `--min-samples 1`, `--min-cluster-size 2` | `--method leaf` |
| Faces in `no_faces/` that should have been found (small/background) | full run with `--min-face 25` | `--det-thresh 0.5`, `--max-side 2400` |
| Junk "people" (posters, statues, blurry crowds) | full run with `--det-thresh 0.7` or `--min-face 60` | |

Change one thing at a time. `merging clusters (similarity X)` lines show how close the merged groups were. Merges at 0.5–0.6 are the ones most likely to be wrong.

## Useful checks (read-only, on the output folder)
- Photos per person: `for d in */; do echo "$(ls "$d" | wc -l) $d"; done | sort -rn`
- Photos in a category: `grep ',unknown$' report.csv`, `grep ',no_faces$' report.csv`
- Each person's `_face.jpg` is the quickest way to spot a wrong group.

## Things to tell the user
- Renamed folders keep their names after re-grouping, but a person merged into another person's group takes that group's name.
- Each run rebuilds all person folders, so they shouldn't keep their own files in them.
