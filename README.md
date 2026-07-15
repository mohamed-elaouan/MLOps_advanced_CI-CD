# Australia Rain Prediction MLOps Application

A deployable MLOps web application that predicts whether it will rain tomorrow in Australia from weather observations. The project combines a trained XGBoost classifier, a Flask prediction interface, Docker packaging, and CI/CD deployment configuration for Google Cloud Platform.

## Project Value

This project demonstrates how a machine learning model can move from training artifacts into a production-style web service. It is designed as a portfolio project for recruiters, hiring managers, and technical reviewers who want to see practical MLOps skills: data processing, model training, model serialization, application serving, containerization, and cloud deployment.

Weather prediction is a practical decision-support use case for logistics, agriculture, transport, event planning, and public services. The app converts daily weather measurements into a simple binary prediction: whether rain is expected tomorrow.

## What the Application Does

The Flask application serves a responsive web UI where users submit the numeric feature vector expected by the trained model. The app loads the serialized model artifact from `artifacts/models/model.pkl`, runs inference, and returns one of two outcomes:

- `YES`: rain is predicted tomorrow.
- `NO`: rain is not predicted tomorrow.

The current model contract uses numeric values, including encoded categorical features such as location, wind direction, and rain-today status.

## Architecture

```text
Raw Weather Data
      |
      v
Data Processing
- date feature extraction
- missing value handling
- label encoding
- train/test split
      |
      v
Model Training
- XGBoost classifier
- evaluation metrics
- model serialization with joblib
      |
      v
Flask Web Application
- loads trained model
- accepts weather features
- returns rain prediction
      |
      v
GCP Deployment Path
- Docker image
- Artifact Registry
- Google Kubernetes Engine
- LoadBalancer service
```

## MLOps Workflow

The repository is organized around a complete machine learning delivery flow:

1. Load weather data from `artifacts/raw/data.csv`.
2. Transform the `Date` field into `Year`, `Month`, and `Day`.
3. Fill missing numerical values with column means.
4. Encode categorical variables for model training.
5. Split the dataset into training and test sets.
6. Train an `XGBClassifier` model.
7. Evaluate accuracy, precision, recall, and F1 score.
8. Save the trained model with `joblib`.
9. Serve predictions through a Flask web application.
10. Package and deploy the app with Docker and Kubernetes on GCP.

## Google Cloud Platform Deployment

This project is prepared for deployment to Google Cloud Platform using:

- **Artifact Registry** for storing Docker images.
- **Google Kubernetes Engine (GKE)** for running the Flask prediction service.
- **Kubernetes Deployment** with 2 replicas for the application workload.
- **Kubernetes LoadBalancer Service** to expose the app publicly.
- **CI/CD pipelines** for automated build, push, and deploy steps.

The deployment manifest is defined in `kubernetes-deployment.yaml`. It creates:

- `Deployment`: `mlops-app`
- `Service`: `mlops-service`
- Container port: `5000`
- Public service port: `80`
- Service type: `LoadBalancer`

### CI/CD Options

The repository includes multiple CI/CD configurations for GCP deployment:

- `.github/workflows/deploy.yml`: GitHub Actions workflow for building the image, pushing it to Artifact Registry, and deploying to GKE.
- `.gitlab-ci.yml`: GitLab CI pipeline with checkout, Docker build/push, and GKE deployment stages.
- `.circleci/config.yml`: CircleCI workflow for Docker image build/push and Kubernetes deployment.

Typical pipeline flow:

```text
Push to repository
      |
      v
Authenticate to GCP
      |
      v
Build Docker image
      |
      v
Push image to Artifact Registry
      |
      v
Get GKE cluster credentials
      |
      v
Apply Kubernetes manifest
      |
      v
Expose Flask app through LoadBalancer
```

### Required GCP Configuration

Before deploying, the cloud environment should include:

- A GCP project.
- Artifact Registry repository in the target region.
- A GKE cluster.
- A service account with permissions for Artifact Registry, GKE, and Kubernetes deployment.
- CI/CD secrets for authentication, such as `GCP_SA_KEY`, `GCP_PROJECT_ID`, `GOOGLE_PROJECT_ID`, `GKE_CLUSTER`, and `GOOGLE_COMPUTE_REGION`, depending on the selected CI provider.

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | Python, Flask |
| Machine Learning | pandas, NumPy, scikit-learn, XGBoost |
| Model Persistence | joblib |
| UI | HTML, CSS, Jinja templates |
| Packaging | Docker, setuptools |
| Cloud Deployment | Google Cloud Platform, Artifact Registry, GKE, Kubernetes |
| CI/CD | GitHub Actions, GitLab CI, CircleCI |

## Project Structure

```text
.
├── application.py                  # Flask app and prediction route
├── src/
│   ├── data_processing.py          # Loading, preprocessing, encoding, splitting
│   ├── model_training.py           # XGBoost training, evaluation, export
│   ├── logger.py                   # Logging helper
│   └── custom_exception.py         # Project exception wrapper
├── pipeline/
│   └── training_pipeline.py        # End-to-end training entrypoint
├── templates/
│   └── index.html                  # Prediction web interface
├── static/
│   └── style.css                   # Responsive UI styling
├── artifacts/
│   ├── raw/                        # Raw dataset location
│   ├── processed/                  # Serialized train/test data
│   └── models/                     # Trained model artifact
├── Dockerfile                      # Container image definition
├── kubernetes-deployment.yaml      # GKE deployment and LoadBalancer service
├── .github/workflows/deploy.yml    # GitHub Actions GCP deployment workflow
├── .gitlab-ci.yml                  # GitLab CI GCP deployment pipeline
├── .circleci/config.yml            # CircleCI GCP deployment workflow
├── requirements.txt                # Python dependencies
└── setup.py                        # Editable package installation
```

## Run Locally

Install dependencies from the project root:

```bash
pip install -e .
```

Train or refresh model artifacts:

```bash
python pipeline/training_pipeline.py
```

Start the Flask application:

```bash
python application.py
```

Open the app:

```text
http://localhost:5000
```

## Run with Docker

Build the image:

```bash
docker build -t australia-rain-mlops-app .
```

Run the container:

```bash
docker run -p 5000:5000 australia-rain-mlops-app
```

## Deploy to GCP Manually

Build and tag the image for Artifact Registry:

```bash
docker build -t us-central1-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/mlops-app:latest .
```

Push the image:

```bash
docker push us-central1-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/mlops-app:latest
```

Connect to the GKE cluster:

```bash
gcloud container clusters get-credentials <CLUSTER_NAME> --region <REGION> --project <PROJECT_ID>
```

Apply the Kubernetes deployment:

```bash
kubectl apply -f kubernetes-deployment.yaml
```

Check the external service address:

```bash
kubectl get service mlops-service
```

## Portfolio Highlights

- Built an end-to-end ML workflow from raw data to deployed prediction service.
- Structured the project with reusable preprocessing and training modules.
- Serialized the trained model and served it through a Flask web application.
- Designed a recruiter-friendly web UI for live model interaction.
- Packaged the app with Docker for reproducible deployment.
- Prepared Kubernetes manifests for running the service on GKE.
- Added CI/CD workflows for automated deployment to Google Cloud Platform.

## Future Improvements

- Persist fitted label encoders so the UI can accept natural category names.
- Replace encoded categorical inputs with dropdowns.
- Add a REST API endpoint for programmatic predictions.
- Add automated tests for preprocessing, training, and Flask routes.
- Add experiment tracking and model versioning.
- Add model monitoring for data drift and prediction quality.

## Author

Mohamed EL Aouan
