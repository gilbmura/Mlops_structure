# models/ — model registry

```
registry.json                 active version, activation history (for rollback), version summaries
versions/vNNN/model.keras     native Keras artifact (served by the API)
versions/vNNN/model.h5        HDF5 artifact (assignment deliverable; reload-tested against model.keras)
versions/vNNN/metadata.json   class order, preprocessing spec, training config/seed, dataset version,
                              validation metrics, artifact SHA-256s, reload check
versions/vNNN/history.json    per-epoch training/validation loss, accuracy, macro-F1
versions/vNNN/input_gate.npz  per-model calibration of the "does this look like a training MRI?" check
```

**Delivery.** `models/brain_mri_classifier.h5` is a copy of the served model's file
(`versions/v002/model.h5`, identical checksum), placed at the top level so it is easy to find. It is not
updated automatically: if a later model is promoted, copy that version's `model.h5` over it. The versions
are small (≤ 3.7 MB per artifact), so their folders and `registry.json` are included as well:

| Version | What it is | Status |
|---|---|---|
| `v001` | baseline CNN (4,932 parameters) | candidate (comparison only) |
| `v002` | final CNN, initial training run | **active (served)** |
| `v003` | retrained from scratch by a worker job on train + 100 admitted uploads | rejected by the promotion rule (meningioma recall −0.053); briefly activated manually to demonstrate hot swap, then rolled back |

Candidates created by later retraining runs stay local (git-ignored). To
reproduce from scratch, delete this folder's contents except this README and run
`python scripts/train_model.py --architecture baseline` then `python scripts/train_model.py --activate`.

**Safety.** Versions appear only after both artifacts were saved *and* reloaded with matching
predictions. Promotion and rollback only switch the `active` pointer in `registry.json`;
earlier versions are never overwritten. The application never loads model files supplied by users.
