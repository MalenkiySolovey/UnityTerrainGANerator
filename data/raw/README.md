# Raw Dataset

Place the local terrain dataset here when training. Raw datasets are usually too
large for the repository and are ignored by Git.

Expected local layout:

```text
data/raw/archive/_dataset/
```

The split files referenced by `configs/train_terrain_gan.yaml` are expected in:

```text
data/splits/train.csv
data/splits/val.csv
```
