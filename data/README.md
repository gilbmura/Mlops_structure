# data/

The **training and test images are included** (`train/`: 4,020 images, `test/`: 868 images,
about 121 MB; unchanged copies from the BRISC 2025 archive, CC BY 4.0 — see the main README for
attribution). Everything else except `manifests/` is regenerated from the original ZIP by:

```powershell
python scripts/prepare_dataset.py --archive "C:\path\to\brisc2025.zip"
```

| Path | Contents | In Git |
|---|---|---|
| `manifests/samples.csv` | every classification file: checksum, decoded-pixel hash, pHash, status, rejection reason, duplicate/group id, split | yes |
| `manifests/{train,validation,test,retrain_demo}.csv` | split membership (paths relative to `data/`) | yes |
| `manifests/exact_duplicate_groups.csv` | identical-pixel groups, labels, whether they crossed the original split | yes |
| `manifests/similar_pairs.csv` | near-duplicate / related / review pairs with Hamming distance and correlation | yes |
| `manifests/dataset_card.json` | provenance (archive SHA-256), counts, config, seed, limitations | yes |
| `train/`, `test/` | original image bytes copied from the ZIP, by class | yes |
| `validation/`, `retrain_demo/` | original image bytes copied from the ZIP, by class | no (regenerated) |
| `cache/` | preprocessed arrays and the similarity index used to screen uploads | no |
| `runtime/` | local SQLite database, staged uploads, job logs (Docker uses a named volume instead) | no |

`retrain_demo/` holds 25 real images per class reserved **before** initial training, so the
upload → review → retrain workflow can be demonstrated with genuine unseen data.
