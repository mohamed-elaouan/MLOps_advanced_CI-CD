# Australia Rain Prediction — MLOps Project

An end-to-end machine learning application that predicts whether it will rain tomorrow in Australia from daily weather observations. It demonstrates the path from raw data and feature preparation to a trained model, an interactive Flask web application, containerisation, Kubernetes deployment, and CI/CD configuration.

**Built as a portfolio project by Mohamed EL Aouan.**

## Executive summary

Weather forecasts support operational planning in sectors such as agriculture, transport, logistics, events, and public services. This project turns current-day weather measurements into a simple decision-support signal: **is rain expected tomorrow — Yes or No?**

The repository is deliberately structured as a production-style MLOps project rather than a notebook-only model. It separates preprocessing, training, application serving, packaging, and deployment configuration so that each part can be reviewed, run, and evolved independently.

| Capability | Implementation |
| --- | --- |
| Prediction task | Binary classification: `RainTomorrow` (Yes / No) |
| Model | XGBoost `XGBClassifier` |
| Input | 24 weather, location, and date features |
| Application | Flask web interface with server-side validation |
| Packaging | Docker |
| Deployment target | Google Kubernetes Engine (GKE) |
| Delivery automation | GitHub Actions, GitLab CI, and CircleCI configuration |

## What this project demonstrates

- Building a repeatable ML pipeline from raw CSV data to a persisted model artefact.
- Preparing data by extracting date features, imputing numerical missing values, encoding categorical fields, and creating a reproducible train/test split.
- Training and evaluating an XGBoost classifier with accuracy, precision, recall, and F1-score metrics.
- Serving the trained model through a responsive Flask interface.
- Shipping the application in a Docker image and deploying it as a replicated Kubernetes workload behind a load balancer.
- Defining CI/CD delivery paths for Google Cloud Platform.

## Results

The committed model artefact was trained with a fixed `random_state=42` train/test split. Its recorded evaluation results are:

| Metric | Score |
| --- | ---: |
| Training accuracy | 89.85% |
| Test accuracy | 86.31% |
| Weighted precision | 85.60% |
| Weighted recall | 86.31% |
| Weighted F1-score | 85.62% |

These results provide a baseline demonstration of the delivery workflow; they should be re-evaluated whenever the data, feature engineering, or model configuration changes.

## Solution architecture

```text
Australia weather data (CSV)
            |
            v
Data processing
  • date decomposition (year, month, day)
  • numerical missing-value imputation
  • categorical label encoding
  • reproducible train/test split
            |
            v
XGBoost training and evaluation
            |
            v
Model artefact (joblib / model.pkl)
            |
            v
Flask prediction application
            |
            v
Docker image → Artifact Registry → GKE Deployment → LoadBalancer
```

## Application experience

The Flask application presents the 24 required features in four clear groups: location and date, temperature and rain, wind conditions, and atmospheric conditions. On submission, it validates the numeric feature vector, invokes the loaded model, and returns a clear `YES` or `NO` prediction.

> **Current model contract:** location, wind directions, and `RainToday` must be supplied as the numeric values produced during label encoding. The encoding mappings are written to the training logs. A future enhancement would persist those encoders and expose user-friendly dropdowns.

## Repository structure

```text
.
├── application.py                  # Flask application and prediction route
├── pipeline/
│   └── training_pipeline.py        # End-to-end processing and training entry point
├── src/
│   ├── data_processing.py          # Ingestion, preprocessing, encoding, splitting
│   ├── model_training.py           # XGBoost training, metrics, model export
│   ├── logger.py                   # Application logging
│   └── custom_exception.py         # Project-specific error wrapper
├── artifacts/
│   ├── raw/data.csv                # Source weather data
│   ├── processed/                  # Persisted train/test datasets
│   └── models/model.pkl            # Trained model used by the app
├── templates/index.html            # Prediction interface
├── static/style.css                # Responsive UI styling
├── Dockerfile                      # Container definition
├── kubernetes-deployment.yaml      # GKE deployment and LoadBalancer service
├── .github/workflows/deploy.yml    # GitHub Actions deployment workflow
├── .gitlab-ci.yml                  # GitLab CI pipeline
├── .circleci/config.yml            # CircleCI pipeline
├── requirements.txt                # Python dependencies
└── setup.py                        # Package metadata
```

## Run locally

### Prerequisites

- Python 3.9 or later
- `pip`

From the repository root, create and activate a virtual environment if desired, then install the project:

```bash
pip install -e .
```

The repository contains processed artefacts and a trained model. To rebuild them from the raw dataset:

```bash
python pipeline/training_pipeline.py
```

Start the web application:

```bash
python application.py
```

Then visit [http://localhost:5000](http://localhost:5000).

## Run with Docker

Build and run the service locally:

```bash
docker build -t australia-rain-mlops-app .
docker run --rm -p 5000:5000 australia-rain-mlops-app
```

Open [http://localhost:5000](http://localhost:5000) in a browser.

## Deployment on Google Cloud

The included Kubernetes manifest defines:

- A deployment named `mlops-app` with two replicas.
- A container listening on port `5000`.
- A `LoadBalancer` service named `mlops-service` exposing port `80`.

The repository also contains GitHub Actions, GitLab CI, and CircleCI definitions for the following delivery sequence:

```text
Repository push
    → authenticate to GCP
    → build Docker image
    → push to Artifact Registry
    → retrieve GKE credentials
    → apply Kubernetes manifest
```

Before using a workflow, update the GCP project, region, Artifact Registry repository, cluster name, image reference, and CI secrets for your environment. The required service account needs permissions to push images and deploy to the target GKE cluster.

For a manual deployment:

```bash
docker build -t us-central1-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/mlops-app:latest .
docker push us-central1-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/mlops-app:latest

gcloud container clusters get-credentials <CLUSTER_NAME> \
  --region <REGION> --project <PROJECT_ID>
kubectl apply -f kubernetes-deployment.yaml
kubectl get service mlops-service
```

## Technology stack

| Area | Tools |
| --- | --- |
| Language and backend | Python, Flask |
| Data and ML | pandas, NumPy, scikit-learn, XGBoost |
| Model persistence | joblib |
| Front end | HTML, CSS, Jinja templates |
| Packaging | Docker, setuptools |
| Cloud and orchestration | Google Cloud Platform, Artifact Registry, GKE, Kubernetes |
| CI/CD | GitHub Actions, GitLab CI, CircleCI |

## Next improvements

- Persist fitted encoders and present categorical inputs as readable dropdowns.
- Add a JSON REST API for programmatic predictions.
- Add unit and integration tests for preprocessing, training, and the Flask route.
- Introduce experiment tracking, model versioning, and a model registry.
- Add monitoring for service health, input drift, and prediction quality.
- Parameterise infrastructure and pipeline configuration for safer multi-environment deployments.

## Author

**Mohamed EL Aouan**
