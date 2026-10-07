Built a point-in-time-correct SQL feature pipeline and LightGBM fraud model on 3000000 simulated transactions, reaching 0.8129 PR-AUC (2.75x a rules baseline) on out-of-time weeks.
Served the model through FastAPI with SHAP reason codes at 3.97 ms p95 latency under 50 RPS, and caught 0.713 of out-of-time simulated fraud in the top 1% of daily scores.
Added a streaming scoring path to the fraud model with batch-identical scores at 858670.9 events/s and a drift alert that fired only on the shifted week (week 11 peak feature PSI 20.9958).
Wrote a CI pipeline with lint, tests, and a model-quality gate that blocks promotion when PR-AUC or recall at 1% regresses, and defined Kubernetes Deployment, Service, and HPA manifests validated against the schema, with Prometheus latency and drift alert rules.

Skills/tools: Python, SQL, DuckDB, LightGBM, scikit-learn, MLflow, FastAPI, SHAP, Kafka, Prometheus, Docker, Kubernetes, GitHub Actions, pytest.

This project simulates 12 weeks of card payments with card testing, account takeover, mule merchants, and shared-device fraud. It scores those payments with point-in-time SQL features and LightGBM, then replays the test weeks through an in-process stream, a CI model gate, and schema-checked Kubernetes manifests.

GitHub: Streaming fraud scoring with batch-identical scores, a CI model gate, and schema-checked Kubernetes manifests on simulated payments.
