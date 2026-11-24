# QA readiness — 2026-11-24

## Verdict

The local application is ready for an Azure update and a rehearsal. The final cloud demonstration
is not signed off: Azure was confirmed deallocated and was left stopped at the owner's request;
the latest changes have not been deployed there,
and no cloud load-test/evaluation evidence was found in this checkout.

## Verified

- Automated regression suite: **125 passed**, including real-model inference, archive validation,
  split leakage, preprocessing, registry/rollback, upload/admission, batch selection, UI button
  state and synthetic worker retraining. One demo-setup test was removed with its deleted
  helper; all remaining tests passed after cleanup. Evidence: `reports/qa_tests.xml`.
- Local Docker: API and Streamlit health checks passed; active model `v002`.
- The separate local recording stack and its temporary data were removed at the owner's
  request. The main local app, models and original reserved images remain. The video will
  use the hosted Azure app; its upload history still needs checking after the VM is started.
- Animated UI update: all four views rendered against the live local API without errors.
  A real headless Chrome check passed desktop and 390 px mobile layout, system reduced motion,
  keyboard operation of the motion switch, prediction and preserving results after that switch.
  Browser evidence and screenshots are local scratch files under `data/cache/ui-qa/`.
  The main local UI was rebuilt; the Azure UI still needs this update.
- Initial training: an isolated Docker stack started with an empty registry. The UI rendered,
  the API queued a baseline job, and the worker completed two epochs on 96 prepared training
  images from random weights. Evaluation used all 868 validation images. The candidate stayed
  inactive and activation was correctly refused with HTTP 409 because this was a quick check.
  This verifies the workflow, not full-training quality. Evidence: `reports/initial_training_demo.json`.
- All four Streamlit views rendered against the live local API with no UI exceptions or error
  messages. The current-batch action was disabled while the existing-pool action remained available.
- Dependency consistency: `pip check` passed. Local and Azure Compose configurations validated.
- Notebook: all 24 code cells have execution counts; no stored error outputs. The entire notebook
  was inspected for execution status, not re-executed during QA.
- Real reserved-image upload, review/admission and separate worker: one epoch on 4,021 images;
  succeeded in an isolated registry. Candidate validation macro-F1 0.8885 versus active 0.9302,
  correctly not promoted. Parent artifact checksum unchanged. See
  `reports/retraining_demo/saved_weights_check.json` and `saved_weights_worker.log`.
- Brief current local Locust check: 10-user target, 55 requests, zero HTTP failures, median 58 ms,
  p95 120 ms. This is a smoke check, not a capacity claim. Raw CSVs: `loadtest/tmp/qa/`.
  Existing longer 1/2/3-container results remain in `loadtest/results/`; they are local results.
- Existing final-test report: 868 images, accuracy 0.8882, macro-F1 0.8952. These are classifier
  results, not proof of clinical accuracy. A four-image live spot check produced two correct and
  two incorrect classifications; show limitations honestly in the video.

## Fixes made during QA

- Enforce the retraining ZIP's file-count limit before decompression, avoiding unnecessary memory
  use when a full dataset is uploaded.
- Convert oversized image-header errors to validation errors; validate images before UI preview.
- Prevent the deployment-check script from silently retraining the existing pool when its upload
  was rejected. It now selects the actual admitted batch.
- Include `v004` in allowed repository artifacts because the current registry references it.
  No served model or existing upload pool was reset.
- Correct outdated Azure/README/video status text and retain the Azure Compose override when scaling.
- Set the Azure worker's default CPU quota to one CPU instead of the local four-CPU default.
- Add initial-training UI/API controls, one shared training/retraining job slot, and an activation
  guard for quick-check candidates. Allow the UI and worker to start before a model is active.
- Add scoped presentation components, accessible animations and live job feedback. Job status
  remains based on stored worker records; no artificial progress percentages are displayed.

## Required before the final video/submission

1. Restore Azure reachability and deploy the current code. Check `/`, `/api/health`, `/api/docs`,
   prediction, uploads, batch-specific retraining and availability on the deployed URL.
   Azure CLI was found outside PATH and confirmed `PowerState/deallocated` on 2026-11-24.
2. Capture production evaluation and Locust results with different container counts; do not
   present local measurements as Azure measurements.
3. Record using Azure. Run the read-only reserved-image check in `docs/demo_video_plan.md`
   before recording uploads. Local admission of the 100 reserved images does not prove they
   were admitted on Azure. Preserve upload history and duplicate/leakage protection.
4. Use small correctly labelled batches. Limits remain 320 MiB ZIP, 640 MiB expanded,
   16,000 archive entries, 200 files per batch and 10 MiB per image.
5. Keep a completed full retraining result for the validation/promotion explanation. Quick demo
   runs use a subset and never promote. Use the notebook to explain at least three image features.
6. Publish all local source/test/artifact changes, confirm registry artifacts are included, rebuild
   the deployment/submission package and add the YouTube URL. The old `deploy-package.tar.gz`
   predates these fixes and should not be reused.
7. Demonstrate both **Train a new model from scratch** and upload-triggered **retraining** in
   the UI. Full initial-training candidates require manual activation after validation review;
   quick initial-training candidates cannot be activated.

## Limits of this audit

This review covered source modules and configuration, regression tests, rendered UI views,
local HTTP prediction, a brief workload, packaging consistency and isolated real-data retraining.
It is not an exhaustive security penetration test, a clinical validation, a cloud performance test
or a guarantee that every possible input is handled. No full 320 MiB upload was load-tested.
