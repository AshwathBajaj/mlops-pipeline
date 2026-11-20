# PayGuard: interview explainer

Ashwath Bajaj. Payment-card fraud detection on the public ULB credit-card table, from a checked download to a served model.

This note is the walk-through I would give in an interview. Every metric is from `reports/metrics.json`, produced by `python -m payguard.pipeline --config configs/train.yaml` on 20 November 2026. That command loaded OpenML dataset 1597. Model version `xgboost-20261120075253`. MLflow run `5055c3c29a1b4f8d8841f1551c633a3e`, registry `payguard-fraud-detector` version 1.

The CI sample and the synthetic generator are not this run. They are named wherever they matter.

## Elevator pitch

PayGuard ranks card transactions so a reviewer only sees a small queue. The training table is the public ULB credit-card set: 284,807 European transactions from September 2013, 492 of them fraud, licence Public, loaded from OpenML 1597. XGBoost had the best out-of-fold PR-AUC, 0.834, and a held-out PR-AUC of 0.850 against a prevalence baseline of 0.00172. A threshold frozen for 50% out-of-fold precision catches 88.8% of test fraud (87 of 98) while flagging 0.425% of test transactions. Test precision at that frozen threshold is 0.360, so the floor did not fully survive the holdout. FastAPI serves the bundle. MLflow stores the run. PSI on the score is 0.0066 on another draw from the same table and 3.808 after a scripted shift.

## The problem

Most transactions are legitimate. On this file, 284,315 of 284,807 rows are legitimate, so a classifier that says "not fraud" every time is 99.83% accurate and never helps a reviewer. The operational question is what fraction of the flagged queue is fraud, and what fraction of the fraud we still miss.

I treat that as two choices.

- Ranking: which model orders fraud above ordinary spend? That is average precision, PR-AUC.
- The queue: which score is high enough to alert? The rule is the highest recall whose out-of-fold precision is at least 0.50.

The public table is what a bank could publish after a confidentiality pass. `V1`–`V28` are principal components. `Time` and `Amount` were left in the clear. There is no merchant, country, device, or card id. Explanations on this model cannot say "this was a foreign online purchase." They can say which components and whether the amount and the hour pushed the score.

## Dataset

| Item | Value |
| --- | --- |
| Name | creditcard |
| OpenML id | 1597 |
| Page | https://www.openml.org/d/1597 |
| File loaded | https://data.openml.org/datasets/0000/1597/dataset_1597.pq |
| Licence | Public |
| Rows | 284,807 |
| Frauds | 492 |
| Legitimate | 284,315 |
| Fraud rate | 0.001727 |
| Features | Time, Amount, V1-V28 |

Origin: transactions by European cardholders in September 2013, two days, collected by the Machine Learning Group at the Université Libre de Bruxelles with Worldline. Citation: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi, Calibrating Probability with Undersampling for Unbalanced Classification, IEEE CIDM, 2015.

`sklearn.datasets.fetch_openml(..., data_id=1597)` is in the loader as a second attempt. On 20 November 2026 the ARFF mirror failed the md5 published in the OpenML description (expected `178bcf9bb1f31a3dfe12d0e577884add`, the downloaded bytes hashed differently on two tries). The parquet URL is a field on that same OpenML record, the download was 73,231,184 bytes, and the frame had the shape and the class counts above. Those counts are from the file we trained on, not copied from the description and assumed.

Train split: 227,845 rows, fraud rate 0.001729. Test split: 56,962 rows, 98 frauds, fraud rate 0.00172. Split seed 42, stratified.

Offline paths, which did not produce the table in the results section:

- `data/samples/creditcard_ci.csv` is 80 frauds and 1,920 legitimate rows drawn from this file (seed 7). `configs/ci.yaml` trains on it. The fraud rate in that file is 4% because it is stratified for a small job, not because the book looks like that.
- `generate_transactions` in `src/payguard/data/generate.py` draws the same column names from a logistic process so unit tests and a total download failure still run. Its labels are not ULB labels.

## Architecture

```text
OpenML 1597 parquet
        |
        v
normalize Class -> fraud, add transaction_id
        |
        v
Pandera schema  -- reject bad rows
        |
        v
stratified 80/20 split
        |
        v
for each model, for each fold:
    fit preprocessor on fold-train only
    fit model
    score fold-validation
        |
        v
Platt calibrator on out-of-fold scores
threshold on calibrated out-of-fold scores
        |
        v
refit preprocessor and model on all training rows
score the test set once
        |
        +--> MLflow run and registry
        +--> models/model.joblib
                |
                +--> FastAPI /predict /predict/batch /explain
                +--> PSI and KS against another real sample
                        and against a shifted copy
                        |
                        v
                     retrain flag
```

The orchestrator is `run_pipeline` in `src/payguard/pipeline.py`. The full-data config is `configs/train.yaml`. The offline job is `configs/ci.yaml`.

## Stage by stage

### Data

`load_transactions` in `src/payguard/data/load.py` is the switch. `source: openml` downloads the parquet, normalises it, and falls through to `fetch_openml` and then to `fallback_csv` or the generator. `source: csv` reads a local file. `source: synthetic` calls the generator. `normalize_transactions` renames `Class` to `fraud`, coerces the numeric columns, and adds `transaction_id` if the file has none. `Time` is not a row id: many transactions share a timestamp.

`validate_transactions` in `src/payguard/data/validate.py` checks a unique id, `Time >= 0`, `Amount >= 0` (zeros exist in the file), finite `V1`–`V28`, and a 0/1 label. A missing column is a `ValueError`. A bad amount is a Pandera `SchemaError`.

### Features

`TransactionFeatureBuilder` in `src/payguard/features/build.py` adds `log_amount = log1p(Amount)`, `hour = (Time mod 86400) / 3600`, and `is_night` when that hour is at most 5 or at least 23. Nothing in the builder looks at other rows. There is no target encoding and no customer history, because the table does not contain a customer key.

`make_preprocessor` median-imputes and standardises those columns. The scaler is fit inside each fold in `evaluate_candidate` (`src/payguard/training/train.py`), then fit again on the full training frame for the model we ship. The components are already on a PCA scale; `Amount` and `Time` are not, which is why the scaler is there for the linear model. Trees share the same preprocessor so serving has one path.

### Models

`build_model` in `src/payguard/training/models.py`:

- Logistic regression, `class_weight="balanced"`, LBFGS.
- `HistGradientBoostingClassifier`, `class_weight="balanced"`.
- `XGBClassifier` with `scale_pos_weight` equal to negatives over positives on the rows that fit is called with, `tree_method="hist"`.

No SMOTE. Resampling before the split leaks, and synthetic minority rows would change a prevalence we are trying to measure. Weights keep the 284,807 real rows.

`run_training` holds out 20% stratified, then runs 5-fold stratified CV on the training frame. Out-of-fold scores are concatenated. Each candidate is refit on all training rows. Test metrics come from that refit.

### Calibration and the threshold

`fit_platt` and `apply_platt` in `src/payguard/scoring.py` fit a logistic regression on the logit of the out-of-fold score. The map is monotone, so it does not change PR-AUC. It changes Brier score, calibration error, and which probability corresponds to the floor.

`recall_at_precision` in `src/payguard/training/metrics.py` keeps the highest recall whose precision on the calibrated out-of-fold scores is at least 0.50. Ties break toward the higher threshold.

For XGBoost that point is threshold 0.0202, out-of-fold precision 0.501, recall 0.850, 335 true positives and 334 false positives. On the test set the same threshold flags 242 of 56,962 transactions: 87 true positives, 155 false positives, 11 false negatives, 56,709 true negatives. Test precision is 87/242 = 0.360. Test recall is 87/98 = 0.888. The floor was met where we chose the threshold and missed on the confirmation set. I would say that, not "precision is 50%."

### Selection

The selection key is `cv_pr_auc_oof`.

| Model | OOF PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| logistic | 0.746 | 0.724 | 0.973 | 0.878 | 0.497 |
| hist GBM | 0.756 | 0.726 | 0.969 | 0.878 | 0.331 |
| xgboost | 0.834 | 0.850 | 0.982 | 0.888 | 0.360 |

Fold PR-AUC is 0.749 ± 0.053 for logistic regression, 0.776 ± 0.010 for histogram gradient boosting, and 0.837 ± 0.023 for XGBoost. Calibrated test Brier is 0.00062, 0.00071, and 0.00051 in that order.

XGBoost leads on the selection metric and on test PR-AUC. Logistic regression's test precision is the one that stayed nearest the 0.50 floor. I still ship XGBoost, because the selection rule is out-of-fold PR-AUC, and I report the operating-point miss beside it.

Raw versus calibrated test probabilities for XGBoost: Brier 0.00145 to 0.00051, expected calibration error 0.00480 to 0.00043. For logistic regression the same correction is Brier 0.0240 to 0.00062 and ECE 0.070 to 0.00018. Balanced class weights move the linear score much further off the observed rate.

### Tracking and serving

`log_training_run` in `src/payguard/tracking.py` opens one MLflow run per model under `payguard-fraud` on a sqlite store, so the registry works locally. The winner is logged as `PayGuardPyfunc` (`src/payguard/serving/pyfunc.py`) with the joblib bundle as its artifact, then registered.

`create_app` in `src/payguard/serving/app.py` loads that bundle at startup. `POST /predict` validates `Time`, `Amount`, and `V1`–`V28` (`src/payguard/serving/schemas.py`). Unknown fields and a negative amount are HTTP 422. `POST /explain` returns the eight largest SHAP contributions and says they are in margin space.

Checked against the committed bundle: the held-out legitimate row in `reports/sample_request.json` scores 0.000088 and is not flagged. A held-out fraud (Amount 0.01, Time 57007) scores 0.969684 and is flagged.

### Explainability

`global_shap` in `src/payguard/explain/shap_report.py` uses `shap.TreeExplainer` because the winner is XGBoost. On 1,500 training rows the largest mean absolute SHAP values are `V14` 1.651, `V4` 1.523, `V10` 0.669, `V12` 0.649, `V3` 0.392, `V8` 0.363, `V19` 0.317, `V11` 0.313, `V26` 0.276, and `hour` 0.264. I would not invent a story about what `V14` "means." The paper and the file say these are principal components. `hour` is the one engineered field in that top ten, and it is smaller than the leading components.

### Drift

`population_stability_index` in `src/payguard/monitoring/drift.py` bins by reference quantiles. `retrain_decision` flags a retrain when at least two features have PSI >= 0.20, or when the score's PSI is >= 0.10.

`run_drift` does not compare the training rows to the synthetic generator. That comparison would move every feature and the flag would be meaningless. The negative control is another sample of the real test rows: prediction PSI 0.0066, prediction KS 0.0145, no feature over 0.20, retrain false. The positive control is those same rows after `simulate_population_shift` (amounts scaled up, `Time` shifted by 12 hours, fixed offsets on `V1`, `V2`, `V3`, `V4`, and `V14`): prediction PSI 3.808, prediction KS 0.652, flagged features `Time`, `Amount`, `V1`, `V2`, `V3`, `V4`, `V14`, retrain true.

The flag writes a report. It does not fit a new model. The shifted rows have no new labels.

## Metrics in plain words

**PR-AUC.** Average precision of the ranking. A useless model lands near the test fraud rate, 0.00172. XGBoost at 0.850 is 0.850 / 0.00172 ≈ 494 times that baseline, which is what a rare-event ranking looks like when the top of the list is actually enriched. It is not a claim that 85% of transactions are handled correctly.

**ROC-AUC.** The probability a random fraud scores above a random legitimate transaction. 0.982 on the test set for XGBoost. All three models are above 0.97. The gap that decided the winner is in PR-AUC (0.724 to 0.850), not in ROC-AUC.

**Recall at a precision floor.** We asked for precision of at least 0.50 on out-of-fold scores, and got recall 0.850 there. On the test set, precision fell to 0.360 and recall was 0.888. In counts: 87 frauds caught, 11 missed, 155 false alarms. The queue is 242 alerts in 56,962 transactions.

**Brier score.** Mean squared error of the probability. The calibrated XGBoost value is 0.00051. A constant prediction of the base rate is about `0.00172 * (1 - 0.00172)`, also near 0.00172, so 0.00051 is a real improvement in probability error, and it is a different question from ranking.

**Expected calibration error.** Weighted gap between predicted probability and observed fraud rate across bins. XGBoost moves from 0.00480 to 0.00043. Logistic regression moves from 0.070 to 0.00018. I threshold the calibrated score.

**Flag rate.** 0.425% of test transactions. That is the review workload.

## Design trade-offs

I shipped the model with the best out-of-fold PR-AUC even though its test precision missed the floor that logistic regression nearly held. Changing the winner after seeing the test set would spend the confirmation set. The right follow-up is a later cohort or a cost curve, not a second look at these 56,962 rows.

The precision floor is 0.50 rather than 0.90. A 0.90 bar is a different product decision. This run's out-of-fold curve does support precision 0.50 at 85% recall, which is a usable queue (about 0.3% of traffic on the training scores). The test set says that particular cutoff was a bit optimistic. I would not quietly raise the threshold until test precision looks like 0.50.

PCA features are a confidentiality trade. We can publish the model and the metrics. We cannot tell a cardholder which raw field hurt them. If the interview needs a named-feature story, that requires a different dataset, and I would not pretend `V14` is "foreign."

The calibrator is fit on out-of-fold scores from the fold models and applied to a refit. Those two models are not identical. Test ECE of 0.00043 for XGBoost is the check that the transfer was good enough here.

Serving reads a joblib file. The registry answers "what did we train?" The API's `model_version` answers "what scored this request?" If they diverge, the live file wins and the mismatch is an incident.

## How to run and demo it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make train
make test
uvicorn payguard.serving.app:app --port 8000
```

`make train` uses the parquet cache at `data/raw/creditcard.pq` when it is already there. Then:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/predict -H 'content-type: application/json' -d @reports/sample_request.json
curl -s http://127.0.0.1:8000/explain -H 'content-type: application/json' -d @reports/sample_request.json
```

`/docs` is the OpenAPI page. `docker compose up --build` starts MLflow on port 5000 and the API on port 8000. The image was not built in the environment that produced these metrics; the scores above were computed by loading `models/model.joblib` in-process, and the API routes are covered by pytest.

`make ci` trains `configs/ci.yaml` on the 2,000-row sample and writes under `models/ci/` and `reports/ci/`. Rebuild this PDF with `make pdf`.

## Limitations and what I would do next

`V1`–`V28` are not actionable features. A decline explanation on this model is "these components moved the score," which is honest and weak as a customer conversation.

Labels in the file are the research labels, not a stream with delay and reversals. Precision against a later chargeback can be lower than 0.360.

The drift control is a scripted shift of real rows, plus a second real sample as the quiet case. It shows the arithmetic. It does not show a slow fraud-mix change.

The retrain flag does not train. I would only fit a new version on a mature labelled window, and only register it if out-of-fold PR-AUC beats the champion.

The threshold should eventually be a cost, not a round precision number. Eleven missed frauds and 155 false alarms are the inputs to that cost, once someone states what each is worth.

## Interview questions

### 1. Give me the 60-second version.

The data is the public ULB credit-card file, OpenML 1597: 284,807 transactions, 492 frauds, licence Public. Accuracy is a trap at a 0.17% base rate. PayGuard compares logistic regression and two boosting models under stratified cross-validation and ships the best out-of-fold PR-AUC, which is XGBoost at 0.834, with test PR-AUC 0.850 and test ROC-AUC 0.982. The threshold was set for 50% out-of-fold precision and, on the test set, catches 87 of 98 frauds with 155 false alarms, so test precision is 0.360. Platt scaling is in front of the threshold. FastAPI serves the bundle, MLflow records it, and PSI stays at 0.0066 on another real sample and jumps to 3.81 on a shifted one.

### 2. Why is accuracy the wrong metric?

Because 284,315 of the 284,807 rows are legitimate. Predicting "not fraud" scores 99.83% and catches nothing. PR-AUC asks whether the rows at the top of the ranking are fraud. The confusion counts at the operating point say how many alerts a person will work: here, 242 test alerts, 87 of them fraud.

### 3. Why select on PR-AUC rather than ROC-AUC?

ROC-AUC compares a random fraud with a random legitimate transaction across the whole score range. We only act on the top. All three models have test ROC-AUC above 0.97. Test PR-AUC runs from 0.724 to 0.850. That gap is the decision. ROC-AUC stays in the report so I can show both.

### 4. Walk through the operating point with the actual counts.

Floor: precision 0.50 on calibrated out-of-fold scores. XGBoost gets there at threshold 0.0202 with precision 0.501, recall 0.850, 335 true positives and 334 false positives. I freeze 0.0202. On 56,962 test rows the refit model flags 242 transactions. Eighty-seven are fraud, 155 are not, so precision is 0.360. There are 98 frauds in the test set (87 + 11), so recall is 0.888. The floor held on the scores we tuned and slipped on the holdout. I would not describe this as "50% precision in production."

### 5. Why did XGBoost beat logistic regression?

The published features are principal components plus amount and time. A scaled linear model can use each component, and it does: out-of-fold PR-AUC 0.746. It cannot cheaply represent interactions among components. XGBoost's out-of-fold PR-AUC is 0.834, and the fold standard deviation is 0.023 against 0.053 for logistic regression. Histogram gradient boosting landed in between at 0.756. I keep the linear model in the comparison because 0.746 is a serious baseline, not a straw man.

### 6. Why not SMOTE?

Fitting SMOTE before the split leaks across the fold boundary. Even inside the fold, synthetic points change the 0.17% prevalence, so the probabilities stop being probabilities of this stream. The alert rate is a threshold on a ranked score. Class weights up-weight the 492 frauds without inventing rows. Platt scaling then puts the probabilities back on the real base rate. Logistic regression's test ECE drop from 0.070 to 0.00018 is that correction.

### 7. Where does leakage sneak in, and how is it blocked?

Scaling with the test mean, choosing the threshold on the test set, and resampling before the split. `evaluate_candidate` fits the preprocessor on the fold's training rows only. The final preprocessor is fit on the training frame only. The threshold uses out-of-fold training scores. The builder cannot leak a global statistic: `log_amount`, `hour`, and `is_night` are functions of that row. I did not target-encode anything. `Time` is a feature, not an id, including when several rows share a second.

### 8. What did calibration actually change?

Not the ranking. For XGBoost, test Brier moved from 0.00145 to 0.00051 and ECE from 0.00480 to 0.00043. For logistic regression, Brier moved from 0.0240 to 0.00062 and ECE from 0.070 to 0.00018. The threshold 0.0202 is a calibrated probability. The same number on the raw score would flag a different set of people. `predict_calibrated` is the only path the API uses.

### 9. Why fit the calibrator on out-of-fold scores?

In-sample scores are over-confident, and a calibrator fit on them learns that. Out-of-fold scores are predictions on rows the fold model did not train on. I then apply the map to the refit model. Test ECE of 0.00043 is the evidence the transfer worked for XGBoost. If it had not, I would hold out a calibration slice.

### 10. Walk through one scoring request.

FastAPI parses JSON into `Transaction`: `Time`, `Amount`, `V1`–`V28`. A negative amount or an extra field is 422 and never reaches the model. The row is encoded with the training preprocessor, scored, and passed through the Platt model. The flag is `probability >= 0.020246` for this bundle. The response includes the threshold and `xgboost-20261120075253`. `/explain` repeats the score and adds SHAP on the encoded row.

### 11. What is PSI, using your two batches?

PSI compares a new sample with a reference using the reference quantiles. Under about 0.10 is the usual stable band. Those cutoffs are config, not theorems. Another draw from the real test rows has prediction PSI 0.0066. The same rows with amount, time, and a few components shifted have prediction PSI 3.808. Only the second one raises the retrain flag. I do not use the synthetic generator as the "no drift" control. It is a different distribution, so the flag would fire for a boring reason.

### 12. How is a schema failure different from drift?

A schema failure means the row is illegal: negative amount, missing `V14`. We refuse to score it. Drift means the rows are legal and the population moved. We can still score them. The monitor says whether enough inputs, or the score, moved past the configured PSI cuts. The two checks are `validate_transactions` and `build_drift_report`.

### 13. The retrain flag fired. What happens next?

A JSON and markdown report. I do not fit a new model on the shifted batch. Those rows do not bring new labels, and training on old labels plus a distorted `Amount` teaches the wrong conditional. The production version of the flag opens a ticket: take a mature labelled window, rerun the pipeline, and register a version only if out-of-fold PR-AUC beats the champion.

### 14. A customer asks why they were declined. What do you show?

I would not translate a SHAP value into "this feature added 12% to your fraud probability." The values are in log-odds space, and `V14` is a principal component, not a merchant. I would show the calibrated probability next to the threshold, and say the largest contributions came from anonymous components plus, if it is in the top, the hour derived from `Time`. For the held-out fraud I scored, the probability is 0.970 against a threshold of 0.020, so it is not a close call, and I still cannot name the raw field. If the product needs a named reason, this dataset is the wrong source and I would say so.

### 15. Why this dataset, and what will you not claim?

It is public, it needs no Kaggle login, and the row counts are checkable: 284,807 rows, 492 frauds. I will not claim the 0.850 test PR-AUC is what the same code would score on a live issuing book. A live book has adversaries, delayed labels, and the raw fields this file replaced with components. I also will not quote the CI sample's metrics as if they were this table. That file is 2,000 stratified rows so the offline job has enough frauds to train.

### 16. How is the model versioned, and which bytes does production run?

`model_version` is `xgboost-20261120075253`, inside the joblib bundle and in every API response. MLflow run `5055c3c29a1b4f8d8841f1551c633a3e` is registry version 1 of `payguard-fraud-detector`, logged-model URI `models:/m-6f7418a380a840d68e900b534323f376`. Production runs the joblib file. The registry is the history. If they disagree, trust the file that answered the request and treat the disagreement as an incident.

### 17. Why not call MLflow on the request path?

A score should depend on a local file. If the tracking server is down, `/predict` should still return. The pyfunc wrapper exists so the registry entry can be loaded later. `create_app` only loads the bundle.

### 18. How would you set the threshold from money instead of from a precision floor?

Put a cost on a false alarm and a cost on a miss. Sweep the threshold on out-of-fold scores and take the minimum expected cost. The inputs from this test set are 155 false alarms and 11 misses, but I would estimate the curve on out-of-fold scores and only then look once at a later cohort. I would not slide 0.0202 until the test precision prints as 0.50.

### 19. What does the CI job actually prove?

`.github/workflows/ci.yml` installs the package on Python 3.11, runs ruff, runs pytest, and runs `configs/ci.yaml` on `data/samples/creditcard_ci.csv`. Pytest covers the schema, `log_amount` and `hour`, the precision-floor helper, PSI, a tiny three-model fit on the synthetic generator, and the API. The CI training job proves the entry point still runs on real rows offline. It does not recompute the 284,807-row metrics. Those stay in `reports/metrics.json` until someone reruns `configs/train.yaml`.

### 20. What breaks first if this is pointed at a real stream?

The features. This model expects `V1`–`V28` computed with the original PCA, plus `Time` in seconds from a fixed origin. A new feed that has not applied that same projection cannot call `/predict` and get a meaningful score. After that: label delay, and adversaries moving toward slices the components do not separate. Prediction PSI and a drop in precision at the frozen threshold are the two alarms I would watch. They are not the same alarm.

### 21. PSI is quiet but fraud is getting through. What else do you watch?

PSI on the inputs can stay small while `p(fraud | x)` changes. On a labelled mature window I would track precision and recall at the frozen threshold, PR-AUC, and ECE. A drop in precision at threshold 0.0202 with prediction PSI still near 0.006 is concept drift, and it is a retrain even when the feature report is green. The scripted shift in this repo is covariate shift, which PSI is built for.

### 22. Logistic regression holds the precision floor and XGBoost does not. Which do you ship?

XGBoost, under the rule we wrote down: best out-of-fold PR-AUC. Logistic regression's test precision is 0.497 and its test PR-AUC is 0.724. XGBoost's test precision is 0.360 and its test PR-AUC is 0.850. Shipping the linear model because the test operating point looks tidier is using the test set to pick the winner. I would ship XGBoost, put the precision slip in the report, and retune the threshold only on scores that are not this test set.

### 23. Why standardise principal components?

`Amount` and `Time` are on different scales from `V1`–`V28`. Logistic regression needs that correction. Trees do not, and one preprocessor keeps the serving path single. The cost is a few standardised copies of `is_night` and `hour`. That is cheaper than two feature pipelines.

### 24. What would you ask for after the first month of reviews?

Labels on the alerts we showed, including the ones marked legitimate, and a sample of transactions we did not show. Without the second sample I can estimate precision and I cannot estimate recall. I would also ask how often a fraud label is reversed. If reversals are common, 0.360 is an upper bound on precision against the final label, and I would train on the matured label.
