# Workflow CI

Repository ini menjalankan retraining otomatis menggunakan MLflow Project.

## Struktur

- `.github/workflows/train.yml`: workflow CI.
- `MLProject/modelling.py`: entry point training.
- `MLProject/conda.yaml`: environment MLflow Project.
- `MLProject/MLProject`: definisi MLflow Project.
- `MLProject/mobile_jkn_reviews_preprocessing/`: dataset siap latih.
- `MLProject/DockerHub.txt`: target image dan secret Docker Hub.

## Menjalankan Lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```
