# Brain MRI classification — video demonstration script

Prepared from the project code, executed notebook and saved reports on 25 November 2026.

Allow approximately **18–22 minutes**, including screen navigation. Read the quoted paragraphs aloud; the **Show** and **Recording note** paragraphs are directions for you. Replace text in square brackets with what the screen actually shows. This is a recording guide, not a claim that every cloud demonstration has already been completed.

## Before you press Record

- Keep your camera on and check microphone audio: the rubric explicitly includes these.
- Use the hosted application at https://gilbert-brain-mri-demo.southafricanorth.cloudapp.azure.com. Open its `/api/docs` and `/api/health` pages in separate tabs.
- Open the executed `notebook/brain_mri_classifier.ipynb` locally and the project in your editor. Use the existing outputs; do not run the entire notebook while recording.
- Have the saved model, its metadata, evaluation reports and completed retraining job ready. Confirm the model version on Azure before quoting its results. The notebook results below are specifically for **v002**.
- Prepare a labelled test image that you have checked against the deployed model. Also prepare an unrelated image to demonstrate refusal. A successful example illustrates the workflow; the full test-set results establish overall performance.
- Prepare distinct, unused reserved images for the single-image, multiple-image and class-folder ZIP upload demonstrations. The reserved files are in `data/retrain_demo/<class>/`; the existing archive is `data/cache/retrain_demo_upload.zip`. Check Azure's upload history first: the local and Azure databases are separate. Previously admitted images will be treated as duplicates. Do not erase history or use the final test set as new training data.
- Include one invalid image in an otherwise valid demonstration batch to show partial acceptance. A safe example is a small text file with a `.jpg` extension. Use an ordinary ZIP, without nested archives or unsafe paths.
- Enter the admin key before recording and leave it masked. Avoid opening `.env`, keys, credentials or terminal output that reveals them.
- Prepare the Azure load-test evidence for **one, two and three API replicas**, using the same workload and measurement procedure. Only a one-replica Azure HTML report was found during this inspection. The older top-level `1_replicas`, `2_replicas` and `3_replicas` results are local tests; they cannot establish the Azure comparison.
- To preserve credit, prepare results during the existing VM's running window. Avoid running full training alongside load tests. Return to one API replica after the comparison.

## 1. Introduce the project — about 45 seconds

**Show:** The hosted application's Predict page, with the public Azure address visible.

> Hello, my name is Gilbert. This project demonstrates the complete machine learning cycle: acquiring image data, cleaning it, training and evaluating a model, deploying it, monitoring it, and retraining it with reviewed uploads.
>
> The application classifies a single brain MRI slice into four dataset classes: glioma, meningioma, pituitary tumour, or no tumour. When the predicted class is a tumour class, the interface also identifies that subtype. Both messages come from the same four-class prediction.
>
> This is an educational image-classification prototype. Its output describes the uploaded slice; it is not a diagnosis of a person. I built the convolutional models myself and initially trained them from random weights, without an externally pretrained model or transfer learning.

## 2. Explain the dataset and cleaning — about 90 seconds

**Show:** Notebook sections **1. Dataset provenance**, **2. Data acquisition and cleaning**, and **3. Split methodology**. Pause on the cleaning counts, duplicate examples and split counts.

> I used the BRISC dataset. It provides labelled MRI images, so the input is non-tabular image data. Although BRISC also includes segmentation masks, this project uses the classification images and their four class labels.
>
> The supplied classification dataset contained 6,000 images. My acquisition pipeline validates the manifests and checks that each image can be decoded. It also calculates image fingerprints so that duplicates can be detected even when filenames differ.
>
> Cleaning removed 144 exact duplicate copies, leaving 5,856 unique images. There were 98 duplicate groups that crossed the dataset's original training and testing split. Keeping those splits would allow the model to encounter identical image content during training and evaluation.
>
> I therefore created new splits. Exact duplicates are removed, and images linked by the similarity checks stay together in one split. The final counts are 4,020 training images, 868 validation images, 868 test images, and 100 images reserved for the retraining demonstration.
>
> Training teaches the model. Validation guides model selection and retraining decisions. The test set measures the selected model's performance. The reserved images are excluded from initial training so that they can demonstrate the arrival of new labelled data.
>
> The dataset does not provide patient identifiers, so these checks reduce image leakage but cannot guarantee patient-level separation.

**Recording note:** Do not count the segmentation masks or repeated segmentation images as additional unique classification images.

## 3. Interpret at least three image features — about 90 seconds

**Show:** Notebook section **4. Exploratory analysis**, especially the feature plots and source-geometry comparison. Point at each feature as you explain it.

> I explored the training split before interpreting model performance. These plots describe measurable image properties; they are not rules for diagnosing tumours.
>
> The first feature is brightness: the average intensity of foreground pixels. Glioma images have a median foreground brightness of about 68, while the no-tumour class is around 89. This suggests that brightness carries information about the dataset classes, but the distributions overlap.
>
> The second feature is contrast: how much foreground pixel intensities vary. It is generally lower in the glioma images and higher in the no-tumour images. This describes how strongly light and dark areas differ within an image.
>
> The third feature is texture, measured through changes in intensity between neighbouring pixels. Higher values indicate stronger local changes. Texture also differs across classes, but no single feature reliably separates them.
>
> I also examined framing. Pituitary images typically contain a larger foreground fraction and appear more tightly cropped. Image size and shape differ between source collections too.
>
> The story is that the dataset contains differences in brightness, texture and acquisition style. A model could learn these patterns as shortcuts. That is why I discuss source bias and inspect errors instead of relying only on one accuracy score.
>
> These hand-measured features support exploration. The CNN itself receives the processed image pixels and learns its own features.

## 4. Explain preprocessing and model design — about 90 seconds

**Show:** Notebook sections **5. Preprocessing** and **6. Model creation**, including the before/after image and model summary. Briefly open `src/preprocessing.py` and `src/model.py` if useful.

> All images follow the same preprocessing steps: decoding, orientation correction, conversion to greyscale, padding to preserve their aspect ratio, resizing to 128 by 128 pixels, and scaling pixel values to the range zero to one.
>
> The notebook, training pipeline and prediction API reuse these functions. This keeps the image representation consistent between training and deployment. The preprocessing specification is also stored with each model version.
>
> Training batches receive augmentation, including small rotations, horizontal flips, shifts and brightness or contrast changes. Validation and test images do not receive these random augmentations.
>
> I compared a small baseline CNN with a larger custom CNN. The final model has four convolutional blocks, with 16, 32, 64 and 128 filters. Convolutions learn visual patterns; pooling reduces their spatial size. Batch normalisation helps training, and dropout provides regularisation.
>
> Global average pooling keeps the model compact. The final model has 295,188 parameters and four output scores, one for each class. Both architectures started with random weights.

## 5. Explain training and evaluation — about two minutes

**Show:** Notebook sections **7. Training** and **8. Evaluation**. Show the learning curves, baseline comparison, per-class precision/recall/F1 and confusion matrix.

> Training uses the Adam optimiser, a starting learning rate of 0.001, batches of 32 images and a maximum of 40 epochs. Class weights account for the unequal number of training images in each class.
>
> Early stopping monitors validation macro F1 and restores the best weights. The learning rate is reduced when validation loss stops improving. For the saved v002 model, training ran for 24 epochs and the best epoch was 16.
>
> I use several metrics because accuracy alone can hide poor performance on an individual class. Accuracy measures the proportion of correct predictions. Precision measures how often a predicted class is correct. Recall measures how many examples of a class are found. F1 balances precision and recall, and macro F1 gives every class equal weight. Loss also measures the quality of the predicted probability distribution.
>
> On the held-out test set of 868 images, the baseline achieved about 63.25 percent accuracy and 0.630 macro F1. The final CNN achieved 88.82 percent accuracy and 0.895 macro F1. Its test log loss was approximately 0.282.
>
> The final CNN's validation macro F1 was about 0.930. That is a different result on a different split; it should not be presented as the final test score.
>
> The confusion matrix reveals an important weakness: 55 pituitary images were classified as meningioma. The notebook also shows confidently wrong predictions. Confidence is a model score, not proof of correctness.
>
> I selected the model using validation results. The test set is excluded from training and from the automatic decision about whether a retrained candidate should replace the active model.

**Recording note:** The notebook's short executable training example uses a small two-epoch run in a scratch registry. Its saved full-training results and metadata establish the main results above. Explain that distinction if you show that cell; do not present the short run as the training that produced 88.82% accuracy.

## 6. Explain the files and functions — about 90 seconds

**Show:** The project directory in your editor. Open only the few files you are discussing; use the reference at the end for detailed questions.

> The notebook documents the experiment, while reusable Python modules implement the pipeline.
>
> In `src/model.py`, `build_baseline` and `build_cnn` define the two architectures. In `src/preprocessing.py`, `preprocess_bytes` and the shared standardisation functions prepare images consistently.
>
> In `src/training.py`, `train_model` handles fitting, class weights and callbacks. In `src/pipeline.py`, `run_training` coordinates the training run, validation and registration of its outputs.
>
> In `src/prediction.py`, `Predictor.predict_bytes` screens an uploaded image and returns either a prediction or a refusal. In `src/evaluation.py`, `evaluate_model` generates the evaluation metrics.
>
> Upload validation and admission live in `src/ingestion`. Database access lives in `src/storage`. Retraining is implemented in `src/retraining.py`, and `src/promotion.py` decides whether a candidate passes the validation requirements.
>
> The `api` folder exposes these operations through FastAPI. The `ui` folder provides the Streamlit interface. The `worker` folder runs longer tasks in the background. The `utils` folder contains shared image, hashing, logging and file helpers.
>
> Configuration is separated from these functions. This makes it easier to change settings and test individual components without putting the entire application into one large file.

## 7. Demonstrate a prediction on Azure — about one minute

**Show:** Return to the hosted **Predict** page. Upload the prechecked labelled test image and submit it. Point to the predicted class, score and version where shown.

> I am now using the application deployed on Azure. This example has the dataset label [true class]. I upload it here, and the application sends it to the Python prediction API.
>
> The deployed model returns [predicted class], with a score of [displayed score]. For this example, [the prediction matches the dataset label / the prediction differs from the dataset label]. The tumour-status message is derived from the predicted class.
>
> This checks the deployed path from browser upload to API preprocessing, model inference and displayed result. The notebook's full test-set evaluation provides the broader measure of model quality.

**Show:** Submit the prepared unrelated image and display the actual outcome.

**If refused, say:**

> This image is refused, so the application does not assign it one of the tumour classes. The screening combines image checks with distance from features learned by our own CNN. It reduces unsuitable inputs, but it cannot prove that every accepted image is a brain MRI.

**Recording note:** If the unrelated image is accepted, acknowledge the screening limitation. Do not describe a refusal that did not occur. Notebook section **9. Prediction functions** demonstrates the same predictor and includes both correct and incorrect examples, followed by the input-screening analysis.

## 8. Demonstrate uploads, database records and mixed batches — about two minutes

**Show:** Hosted **Uploads & retraining**. Briefly show the single-image and multiple-image controls using distinct labelled samples. Then upload the prepared mixed ZIP and follow its processing status.

> Retraining accepts one image, multiple images or a ZIP containing class folders. For individual images I supply a label. In a ZIP, the recognised folder names provide labels for review.
>
> The labels must belong to the four supported classes. The model's own prediction is not used as the training label.
>
> ZIP processing runs as a background job. The browser receives a job identifier and displays progress while screening continues. A completed screening job does not mean that every image has been accepted.

**Show:** The completed batch's counts and the rejected-file reasons. Switch the review filter to eligible staged images.

> In this batch, [number] files were rejected, [number] were duplicates, and [number] remain eligible for review. A bad image is excluded individually while qualifying images can proceed. If the archive itself fails a safety check, the whole archive may be refused.
>
> Passing screening is not the same as entering training. I review the image content and the supplied labels, confirm the review, and admit only the appropriate images.

**Show:** Admit the reviewed valid images. Show their records through the upload list/API response and, briefly, the `uploads` table definition in `src/storage/database.py`.

> The application saves an upload record in SQLite, including the batch, filename, label, status, image fingerprint and stored path. Accepted image files are kept in persistent storage; the database tracks them and their review state. The image bytes are not stored as database blobs.
>
> The current batch can enable retraining once it contains admitted images and processing is complete. If all images are rejected, retraining with that batch stays disabled. Retraining from an existing admitted pool is a separate action.

**Recording note:** Show the job or batch identifier so the viewer can connect the upload, review and training stages. Do not imply that an older completed job card was created by the currently selected rejected batch. If a mixed-batch result differs from this expected flow, resolve it before the final recording instead of narrating an unobserved success.

## 9. Trigger retraining and explain the outcome — about 90 seconds, plus a labelled time cut

**Show:** The admitted count and **Retrain with this batch**. Start one job, then show its identifier, progress and logs. Show that a prediction still responds while the worker is training.

> This button creates a retraining job for the admitted images in the selected batch. The worker combines those images with the original training split and applies the saved preprocessing rules.
>
> Initial model training started from random weights. Retraining continues the weights of our own saved custom CNN, using a fresh optimiser and a smaller learning rate of 0.0001. No externally pretrained model is introduced.
>
> Training runs outside the prediction request, so the serving API can continue answering requests while the worker trains a candidate. Both processes still share the VM's CPU and memory.

**Show:** The completed job's `parent_version`, `initialization`, `uploads_included`, candidate validation results, promotion decision and active version. If training takes longer than the video, pause recording and resume with the **same job identifier** visible. Label the cut “After retraining completed”.

> The completed job records which model it started from and how many uploaded images it used. The candidate is evaluated on the fixed validation split before it can replace the active model.
>
> The promotion rule allows no more than a 0.01 drop in macro F1 and no more than a 0.05 drop in any class's recall. Both checks must pass. A higher overall score alone is not enough if one class becomes substantially worse.

**If promoted, say:**

> This candidate passed the checks and was activated. The serving API loads the active version from the registry.

**If not promoted, say:**

> Training completed, but this candidate was not activated. The reason shown is [actual decision reason]. The previous model remains active. Completing a training job and promoting its result are separate outcomes.

**If using Quick demo run, say instead:**

> This is a two-epoch run on a small subset to demonstrate the workflow. Quick demo candidates are never promoted. I will show a separately identified full retraining job for the complete validation and promotion evidence.

**Recording note:** Prefer one full job with a time cut over repeatedly training during rehearsals. Do not present a historical local result as an Azure retraining run. Show **Train a new model from scratch** briefly too; explain that it is separate from continuing an existing model, without starting another costly job.

## 10. Explain cloud deployment and monitoring — about one minute

**Show:** Hosted **System status**, `/api/health`, `/api/docs`, and the Azure terminal's Docker Compose service list.

> The application is deployed on an Azure virtual machine using Docker Compose. Caddy provides the public HTTPS entry point. It routes the user interface to Streamlit and API traffic through the nginx gateway to the prediction service.
>
> Separate containers handle the interface, API, background worker, monitoring probe and routing. Persistent storage keeps the dataset, models and upload records across container recreation.
>
> This status page shows the model version, the responding API replica and its process uptime. Periodic health probes record whether the API responds and how long the health request takes.
>
> The availability percentage describes the probes that were actually recorded. Because this probe runs on the same VM, it does not measure an overnight VM shutdown as an external monitoring service would.
>
> The Data and evaluation page displays saved validation and test reports. Those are recorded model evaluations, while these public API requests demonstrate the running deployment. I keep those two kinds of evidence distinct.

**Recording note:** For production prediction evaluation, compare live outputs with known dataset labels and identify the serving model version. `scripts/check_deployment.py` can collect one prediction per class and check service behaviour, but it does not require all four predictions to be correct. A successful service check is not proof of perfect classification accuracy.

## 11. Demonstrate Locust and container scaling — about 90 seconds

**Show:** Locust's browser interface at http://127.0.0.1:8089, with the Azure hostname as the target. Then show the saved Azure results for one, two and three API replicas, their Docker service lists and response replica counts.

> Locust runs on my laptop and sends requests to the Azure deployment. The test task uploads actual MRI images to the public prediction endpoint. Each simulated user waits between requests and repeats the task.
>
> I compare one, two and three API replicas using the same model, images, virtual-user count, spawn rate and measurement procedure. I distinguish the ramp-up period from the measurement period and avoid running retraining during the comparison.
>
> For one API replica, the observed throughput is [value] requests per second, average response time is [value] milliseconds, and the 95th-percentile response time is [value] milliseconds, with [value] failed requests. The two-replica and three-replica reports show [read the corresponding measured values].
>
> The 95th percentile means approximately 95 percent of measured requests finished within that time. The 99th percentile shows more of the slow tail. These are end-to-end times from the laptop, including network and application work, rather than pure model inference time.
>
> The service list establishes how many API containers were running, and the response identifiers show which containers answered. The comparison changes API replicas, not the total count of all supporting containers.
>
> All replicas share one VM, so adding containers does not add physical CPU capacity. I interpret the actual throughput and latency results rather than assuming that three replicas must be fastest.
>
> A successful HTTP response means the request was handled. It does not establish whether the prediction was correct; that requires comparison with a known label.

**Recording note:** Say “one API replica”, not “one container”, when the full stack also contains the UI, worker and proxies. The current load-test task counts HTTP 200 responses as successes; an accepted response may still contain a model refusal. Inspect response meaning separately from transport success. Do not claim Azure scaling evidence is complete until the matching two- and three-replica results exist.

## 12. Close with deliverables — about 30 seconds

**Show:** Repository README, the notebook, saved `.h5` artifact and result folders. Finish on the public application.

> The repository contains the notebook, modular source code, trained model artifacts, deployment configuration and recorded evaluation evidence. The README provides the setup instructions and links to the hosted application and demonstration video.
>
> This demonstration has shown prediction, reviewed uploads, retraining, validation before activation, service monitoring and load testing. The main limitations are dataset source bias, uncertain patient separation and imperfect screening of unfamiliar images.
>
> The project will be submitted as a ZIP of the code repository in the first submission attempt and as the GitHub repository URL in the second attempt. Thank you.

**Recording note:** Add the final YouTube link and the actual cloud comparison results to the README before submission. Verify the saved model is included or accessible as required; a filename in the README is not a substitute for the artifact.

## File and function reference for your explanation

These are short answers to use if you are asked about implementation. You do not need to read every entry in the video.

### Dataset and experimentation

- **`notebook/brain_mri_classifier.ipynb`:** The executed experiment record: provenance, cleaning, splits, feature interpretation, preprocessing, model design, training, evaluation, prediction examples and limitations. It imports reusable project code instead of duplicating the whole application.
- **`scripts/prepare_dataset.py` → `src/dataset/prepare.py::prepare_dataset`:** The command-line entry point and implementation for inspecting the archive, removing duplicates, assigning grouped splits, writing images and producing manifests. `_resolve_exact_duplicates`, `_link_related` and `_assign_splits` separate the main decisions.
- **`data/manifests/`:** Records which images belong to each split and provides the dataset card used to explain cleaning. These records make split membership traceable.
- **`src/features.py::image_features`, `feature_table`, `summarize_by`:** Measure image properties and summarise them for dataset exploration. They explain the plots; they do not replace the image tensor passed to the CNN.
- **`src/preprocessing.py::PreprocessingSpec`, `standardize`, `preprocess_bytes`, `load_images`:** Store and apply the preprocessing contract for uploaded bytes and dataset files. `letterbox` preserves aspect ratio using padding.
- **`src/augmentation.py`:** Adds random transformations to training batches. Evaluation uses deterministic preprocessing.

### Models, training and inference

- **`src/model.py::build_baseline`, `build_cnn`, `build_model`, `compile_model`:** Construct the architectures and configure the optimiser, loss and training metrics. The output class order is defined consistently across the project.
- **`src/training.py::train_model`, `compute_class_weights`:** Fit the model and manage class weighting, callbacks and training history.
- **`src/pipeline.py::load_training_data`, `run_training`:** Load the selected data and coordinate training, validation, metadata and model registration.
- **`src/evaluation.py::evaluate_model`, `classification_report`:** Calculate model-quality metrics, including per-class results and the confusion matrix. These are different from Locust's request-time metrics.
- **`src/prediction.py::Predictor.predict_bytes`:** Decode and screen an input, use the model's preprocessing specification and return a structured prediction or refusal. `ActiveModelProvider.get` supplies the current predictor and checks for active-version changes.
- **`models/versions/v002/model.keras`, `model.h5`, `metadata.json`, `history.json`:** The served Keras artifact, the assignment-compatible HDF5 artifact, the model's settings/evaluation information, and its epoch history. Metadata identifies which results belong to which model.
- **`src/registry.py::ModelRegistry.register`, `activate`, `rollback`:** Save and reload-check versioned artifacts, select a serving version and support rollback. Registration does not itself mean a candidate is active.

### Uploads, storage and retraining

- **`api/routers/uploads.py::upload_images`, `upload_zip`, `admit`:** Expose single/multiple image uploads, ZIP submission and admission through the API. ZIP submission returns a background job rather than waiting for all images to be screened.
- **`src/ingestion/queue.py::queue_zip`:** Stores the submitted archive within configured limits and creates or identifies its processing job.
- **`src/ingestion/service.py::IngestionService.ingest_files`, `screen_archive_entry`, `admit`:** Apply file-level checks, record the outcome and control admission. Decodable content, valid labels, duplicate checks and protection of evaluation data serve different purposes.
- **`src/ingestion/batch.py::screen_archive` and `worker/uploads.py::run_upload`:** Process ZIP entries and persist progress and results. Individual rejections remain visible; incomplete or failed archive processing cannot supply an admitted partial batch.
- **`src/storage/database.py::Database.transaction`:** Provides SQLite transactions. The schema contains upload records, job records and health probes. `UploadStore` and `JobStore` keep database operations out of the UI and model code.
- **`api/routers/jobs.py::request_retraining`:** Validates and queues a retraining request; this endpoint is the programmatic equivalent of the retraining button.
- **`src/retraining.py::run_retraining`:** Loads the active project's own model, checks the selected admitted images, combines them with original training data, trains a candidate and evaluates it. The job records its parent version and included uploads.
- **`src/promotion.py::decide_promotion`:** Applies the validation macro-F1 and per-class recall rules. Smoke runs are separately prevented from activating a model.
- **`worker/main.py::process_one`:** Claims a queued job and runs the appropriate handler outside the API request. Initial training and retraining share controlled job scheduling.

### Interface, deployment and tests

- **`api/main.py`:** Creates the FastAPI application and connects its routers and services. API routes orchestrate the reusable project functions.
- **`ui/app.py`, `ui/views/`, `ui/api_client.py`:** Set up Streamlit navigation, render the pages and call the API. The views cover prediction, insights, status, uploads and training progress.
- **`worker/probe.py::probe_once`:** Calls the health endpoint and stores the outcome and latency for monitoring.
- **`config/pipeline.yaml`:** Holds dataset, preprocessing, training, promotion, upload and screening settings. `config/settings.py` handles environment-specific application settings.
- **`utils/`:** Shared helpers for decoding images, hashing, file handling, CSV export and logging. Keeping these helpers reusable avoids repeated implementations.
- **`docker-compose.yml` and `deploy/azure/docker-compose.azure.yml`:** Define the container services, shared storage and Azure-specific settings. The deployment documentation explains how to run them together.
- **`loadtest/locustfile.py::PredictUser.predict`:** Uploads real images to `/api/predict`. The host is the public Azure origin without an extra `/api` suffix. `write_replica_counts` can save response counts by replica when its output path is configured.
- **`scripts/check_deployment.py`:** Exercises the running health and prediction endpoints and can optionally exercise uploads/retraining. Its default service checks are not a full accuracy evaluation.
- **`tests/`:** Automated checks for pipeline and application behaviour. Explain the test coverage you have actually run; do not quote an old test count as a fresh test result.
- **`README.md`, `reports/`, `loadtest/results/`:** Setup and submission instructions, model/QA evidence and performance reports. Identify whether each report was generated locally or against Azure.

## Five questions to rehearse

**Why did you replace the original splits?**

> Exact duplicates crossed the supplied training and test splits. I removed duplicate copies and kept linked images together to reduce image leakage. Patient separation remains unverified because patient IDs are unavailable.

**Why continue saved weights when you said no transfer learning?**

> I initially trained our own architecture from random weights on this dataset. Retraining continues that project's own learned model. I am not importing another model's pretrained representations.

**Why can a mixed batch retrain when some images were rejected?**

> Eligibility is decided per image after the archive passes its checks. Rejected files remain excluded. Reviewed valid images can be admitted, and a completed batch with admitted images can supply a retraining job.

**Why did training finish without changing the active model?**

> Completing optimisation produces a candidate. Activation requires passing the promotion rules. A quick demo run is never promoted, and a full candidate can be rejected if its validation performance fails a required check.

**Does zero Locust failure mean the model is always correct?**

> No. It measures successful request handling under that workload. Classification correctness requires ground-truth labels and model evaluation metrics. Response content must also be checked to distinguish a prediction from a refusal.
