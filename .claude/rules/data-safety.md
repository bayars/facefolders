---
paths:
  - "sort_faces.py"
  - "compose.yaml"
  - "run.sh"
---

# Protecting the user's photos

- Photos are the user's personal family pictures. **Originals are never modified, moved or
  deleted**: `/photos` is mounted read-only; keep it that way.
- `clear_previous_output` may only delete folders the tool itself created: those containing
  `.person.json`, legacy `person_*` folders, `unknown/`, `no_faces/`. Never widen this to
  "everything in the output dir"; users may point OUTPUT_DIR at a folder with other content.
- Don't break the `.person.json` marker format (`{"centroid": [512 floats]}`) or the
  rename-to-name behaviour; users' manual naming work depends on it. If it must change, read the old
  format too.
- Validate the output dir is writable **before** detection (`check_writable`); failing after a
  long detection pass wastes the user's time.
- Never commit photos, `.env`, output folders or `faces.*` caches (the embeddings are biometric
  data). Real photos never go into the repo or test fixtures; use the synthetic
  InsightFace sample images (see the `smoke-test` skill).
