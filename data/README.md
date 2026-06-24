# Data

This directory is a local mount point for dataset files. Raw and processed data
are intentionally ignored by Git because they are large and environment-specific.

Expected structure:

```text
data/
  raw/
  processed/
  splits/
```

The committed split CSV files in `data/splits/` may be used with the original
dataset layout.
