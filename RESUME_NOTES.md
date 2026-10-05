Built a point-in-time-correct SQL feature pipeline and LightGBM fraud model on 3000000 simulated transactions, reaching 0.8129 PR-AUC (2.75x a rules baseline) on out-of-time weeks.
Served the model through FastAPI with SHAP reason codes at 4.34 ms p95 latency under 50 RPS, with versions promoted through an MLflow model registry.
Added PSI-based drift monitoring that flagged an injected fraud-pattern shift in week 11 (peak feature PSI 20.9958) and did not alert in weeks 10 or 12.
Caught 0.713 of out-of-time simulated fraud when reviewing the top 1% of scores each day, with the review threshold chosen by expected cost on the validation week only.

Skills/tools: Python, SQL, DuckDB, LightGBM, scikit-learn, MLflow, FastAPI, SHAP, pytest.

This project simulates 12 weeks of card payments with card testing, account takeover, mule merchants, and shared-device fraud, plus legitimate lookalikes and label noise. It builds point-in-time SQL features, trains a LightGBM scorer against hand-written rules, serves scores with SHAP reason codes, and checks a later week for population drift.

GitHub: Point-in-time fraud features, a cost-thresholded LightGBM scorer, FastAPI serving, and PSI drift monitoring on simulated payments.
