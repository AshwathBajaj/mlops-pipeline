# PayGuard: interview explainer

Ashwath Bajaj. Payment-card fraud detection, from a checked table to a served model.

This note is the walk-through I would give in an interview. Every metric is from `reports/metrics.json`, produced by `python -m payguard.pipeline --config configs/train.yaml` with seed 42 on 20 November 2026. Model version `logistic_regression-20261120071250`. MLflow run `49fe7559f07b43e9abd657a0a85a3f5e`, registry `payguard-fraud-detector` version 1.

## Elevator pitch

PayGuard ranks card transactions so a reviewer only sees a small queue. On 30,000 synthetic payments with a 2.14% fraud rate, logistic regression had the best out-of-fold PR-AUC, 0.223, and a held-out PR-AUC of 0.283 against a prevalence baseline of 0.0215. The shipped threshold keeps test precision at 0.407 and catches 17.1% of test fraud while flagging 0.9% of transactions. The same bundle is what the FastAPI service loads. MLflow stores the run. A PSI check tells us when a new batch is no longer the population we trained on.

## The problem

Most transactions are legitimate. A classifier that says "not fraud" every time is about 97.9% accurate here and never helps a reviewer. The operational question is: of the transactions we put in front of a person, what fraction are actually fraud, and what fraction of the fraud do we still miss?

I treat that as two separate choices.

- Ranking: which model orders fraud above ordinary spend? That is average precision, PR-AUC.
- The queue: which score is high enough to alert? That is a threshold chosen so precision stays at or above 0.40, with recall as high as that floor allows.

Labels in a real issuer feed arrive late and are noisy. This repository does not pretend to have that feed. It uses a documented generator so the pipeline, the tests, and CI run without credentials, and so an explanation can be checked against a known process.

## Architecture

```text
generator or CSV or URL
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
                +--> PSI and KS against a new batch
                        |
                        v
                     retrain flag
```

The orchestrator is `run_pipeline` in `src/payguard/pipeline.py`. Config lives in `configs/train.yaml`. The smaller job CI runs is `configs/ci.yaml`.

## Stage by stage

### Data

`generate_transactions` in `src/payguard/data/generate.py` builds the table. Amounts are log-normal with a rare spike. Hour, account age, utilisation, distance, merchant risk, and the categorical fields (merchant category, channel, country match, device trust) have fixed distributions. `fraud_effects` is the linear predictor: large log-amount, foreign, online, a new device, night, high velocity, a high amount-to-average ratio, far from home, risky merchants, plus three interactions (foreign and online, new device at night, high ratio and online). An intercept search holds the mean probability near the requested rate, here 0.02. Labels are Bernoulli draws plus a 0.2% flip, so the process is not perfectly separable. The realised fraud rate on this run is 0.021433.

`load_transactions` in `src/payguard/data/load.py` supports three sources. `synthetic` calls the generator. `csv` reads a file. `url` downloads a CSV and, on a network error, falls back to the generator. CI uses `synthetic` on purpose.

`validate_transactions` in `src/payguard/data/validate.py` runs a Pandera schema: positive amount, hour 0-23, utilisation and merchant risk in [0, 1], categories from the closed sets in `src/payguard/schema.py`, unique transaction ids, and a 0/1 label. A missing column raises `ValueError`. A bad value raises `SchemaError`. Drift is not this function's job.

This run: 30,000 rows, 24,000 train, 6,000 test, train fraud rate 0.021417, test fraud rate 0.0215.

### Features

`TransactionFeatureBuilder` in `src/payguard/features/build.py` adds only row-wise fields: `log_amount`, `amount_to_avg_ratio`, `is_night`, `is_high_velocity`, `is_foreign_online`, `is_new_device_night`. It does not look at other rows, so it cannot leak a target mean.

`make_preprocessor` chains that builder with a `ColumnTransformer`. Numeric columns are median-imputed and standardised. Categorical columns are mode-imputed and one-hot encoded with `handle_unknown="ignore"`. Those steps are fit inside each cross-validation fold, in `evaluate_candidate` in `src/payguard/training/train.py`, and fit again on the full training frame before the model we ship.

### Models and imbalance

`build_model` in `src/payguard/training/models.py` builds three candidates.

- Logistic regression, `class_weight="balanced"`, LBFGS. This is the baseline.
- `HistGradientBoostingClassifier`, `class_weight="balanced"`.
- `XGBClassifier` with `scale_pos_weight` equal to negatives divided by positives on the rows that model is fit on, `tree_method="hist"`.

I did not use SMOTE. Resampling before the split leaks, and synthetic minority rows distort probabilities. Weights keep every real transaction and the original prevalence. The threshold is what changes the size of the queue.

`run_training` does one stratified holdout (`test_size` 0.2), then `StratifiedKFold` with 5 folds on the training frame. Each fold fits its own preprocessor and model. Out-of-fold scores are concatenated. The model is then refit on all training rows for serving. Test metrics come from that refit, so they are not the same object as the out-of-fold scores. The gap is small and is the usual pattern: cross-validation chooses, the refit is what we deploy, the test set is a single confirmation.

### Calibration and the threshold

`fit_platt` and `apply_platt` in `src/payguard/scoring.py` fit a logistic regression on the logit of the out-of-fold score. The map is monotone, so it does not change who ranks above whom, and it does not change PR-AUC. It changes Brier score, calibration error, and which probability corresponds to the precision floor.

`recall_at_precision` in `src/payguard/training/metrics.py` walks the precision-recall curve of the calibrated out-of-fold scores and keeps the highest recall whose precision is at least the configured floor (0.40). Ties break toward the higher threshold, which flags fewer people. `operating_point` then counts true positives, false positives, false negatives, and true negatives at that frozen threshold on the test set. The test labels are not used to move the threshold.

On this run the floor was achieved. Out of fold: threshold 0.256, precision 0.400, recall 0.167, 86 true positives and 129 false positives. Test: precision 0.407, recall 0.171, flag rate 0.009, 22 true positives, 32 false positives, 107 false negatives, 5,839 true negatives.

### Selection

The selection key is `cv_pr_auc_oof`, the average precision of the concatenated out-of-fold scores. Fold mean and fold standard deviation are reported next to it so a lucky fold is visible.

| Model | OOF PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| logistic | 0.223 | 0.283 | 0.870 | 0.171 | 0.407 |
| hist GBM | 0.156 | 0.238 | 0.823 | 0.147 | 0.475 |
| xgboost | 0.160 | 0.229 | 0.820 | 0.101 | 0.542 |

Fold PR-AUC is 0.230 ± 0.019 for logistic regression, 0.166 ± 0.030 for histogram gradient boosting, and 0.168 ± 0.027 for XGBoost. Calibrated test Brier is 0.018, 0.018, and 0.019 in that same order.

Logistic regression wins on the selection metric and also has the highest test PR-AUC. I do not treat the test column as a second chance to pick a winner. It is there so I can see whether the out-of-fold ranking survived.

Raw versus calibrated probabilities for the winner, on the test set: Brier 0.146 to 0.018, expected calibration error 0.315 to 0.007. `class_weight="balanced"` pushes the raw score up. The reliability plot in `reports/plots/calibration.png` shows those raw scores sitting well above the observed fraud rate. After Platt scaling they sit close to the diagonal, in the low-probability region, which is where a 2% event should sit.

### Tracking and the artifact

`log_training_run` in `src/payguard/tracking.py` opens one MLflow run per model under the experiment `payguard-fraud`, using a sqlite tracking URI so the model registry works locally. It logs parameters, the metric dict, and, for the winner, the plots, the drift report, and a `PayGuardPyfunc` model (`src/payguard/serving/pyfunc.py`) whose artifact is the joblib bundle. That pyfunc is what gets registered.

The bundle itself, written by `run_pipeline`, contains the preprocessor, the estimator, the calibrator, the threshold, encoded feature names, a 100-row background matrix for linear SHAP, the model version, and a copy of the public metrics. FastAPI loads this file. It does not query MLflow when a request arrives.

### Serving

`create_app` in `src/payguard/serving/app.py` loads the bundle in the FastAPI lifespan. `GET /health` returns `ok` when the file loaded and `degraded` when it did not. `POST /predict` and `POST /predict/batch` validate a pydantic `Transaction` (`src/payguard/serving/schemas.py`) and return the calibrated probability, the flag, the threshold, and the model version. Unknown categories and illegal amounts are HTTP 422. A missing file is HTTP 503 on the scoring routes.

`POST /explain` encodes the same row, runs SHAP, and returns the eight largest contributions. The response says the values are in margin space.

Checked against the committed bundle: the sample row in `reports/sample_request.json` scores 0.009044 and is not flagged. An obvious high-risk row (online, foreign, new device, hour 2, gambling, amount far above its 30-day average) scores 0.999745 and is flagged.

### Explainability

`global_shap` and `explain_encoded_row` in `src/payguard/explain/shap_report.py`. Trees use `shap.TreeExplainer`. The linear winner uses `shap.LinearExplainer` on a background of at most 100 encoded training rows. For this run the top mean absolute SHAP values on 1,500 training rows are:

- log_amount 0.538
- channel_online 0.240
- device_trust_trusted 0.120
- txn_count_24h 0.112
- country_match_foreign 0.109
- country_match_domestic 0.093
- is_night 0.088
- utilization 0.087

`log_amount` and `channel_online` leading is what I expect from `fraud_effects`. SHAP here is an explanation of the model, not a proof of causality, but it is a useful check that the model latched onto the process I actually coded.

### Drift and the retrain flag

`population_stability_index` in `src/payguard/monitoring/drift.py` cuts bins at the reference quantiles, clips empty bins at 1e-6, and sums `(actual - expected) * log(actual / expected)`. Categorical PSI uses the category shares. Numeric columns also get a Kolmogorov-Smirnov statistic via `scipy.stats.ks_2samp`.

`retrain_decision` flags a retrain when at least two features have PSI >= 0.20, or when the score's PSI is >= 0.10. Both cutoffs are in `configs/train.yaml`.

`run_drift` scores two controls with the shipped model. A fresh synthetic draw (`seed + 101`, 4,000 rows) against a 4,000-row training reference: prediction PSI 0.004, prediction KS 0.014, retrain false. The same draw after `simulate_population_shift` (amounts scaled up, hour rotated, more online, foreign, and new-device rows, higher velocity, higher merchant risk, larger distance): prediction PSI 6.135, prediction KS 0.814, eight features over the PSI cutoff (amount, hour, txn_count_24h, distance_from_home_km, merchant_risk_score, channel, country_match, device_trust), retrain true.

The flag is a report. Nothing in this repository starts a training job off the back of it. That is deliberate: a retrain on a drifted, unlabelled batch would teach the model the new population's features with the old labels.

## Metrics in plain words

**PR-AUC / average precision.** Walk down the scores from riskiest to safest. At each point, precision is "of the people flagged so far, how many are fraud?" Average precision is the area under that curve. A useless model lands near the fraud rate, 0.0215 on the test set. 0.283 means the ranking is much better than that, about 13 times the baseline, and it is still far from perfect. The curve in `reports/plots/pr_curves.png` falls quickly: precision near 0.40 is available around 20% recall, and precision is near the base rate once recall is high. That shape is why the queue is small.

**ROC-AUC.** The probability a random fraud scores above a random legitimate transaction. 0.870 on the test set for the winner. I report it because people ask. I do not select on it. With a 2% positive class, ROC-AUC can look strong while the top of the queue is still full of false alarms. PR-AUC looks at that top directly.

**Recall at a precision floor.** Precision 0.40 means four in ten alerts are fraud, about 19 times the base rate. Recall 0.171 means we still miss about 83% of fraud. That is an honest description of a high-precision queue, not a claim that we "catch fraud." The misses are the 107 false negatives on the test set.

**Brier score.** The mean squared error of the probability. Lower is better. A model that always predicts the base rate scores about `rate * (1 - rate)`, near 0.021 here. The calibrated winner at 0.018 is a bit better than that constant on probability error, which is a different question from ranking. Ranking is the PR-AUC. Brier is "are the probabilities honest?"

**Expected calibration error.** Split the probabilities into bins. In each bin, compare the average predicted probability with the actual fraud rate, and weight by how many rows fell in the bin. 0.315 before calibration and 0.007 after, on the test set, is the whole reason Platt scaling is in the pipeline. I would not threshold raw balanced-class probabilities.

**Flag rate.** 0.9% of test transactions are alerted. That is the workload. 54 alerts in 6,000 transactions, 22 of them fraud.

## Design trade-offs

I engineered the interactions I believed in, and then the linear model won. That is a result, not a failure of boosting. If I had hidden those interactions, the trees would have had something to split on and might have won, and I would have learned less about whether the features were doing the work. Keeping all three models in the config means the next dataset can pick a different winner without a code change.

I chose a 0.40 precision floor after seeing that a 0.70 floor only held in the extreme tail, on a few dozen training alerts. A threshold estimated from a handful of events will not reproduce. 0.40 is still a concentrated queue (215 out-of-fold alerts, precision exactly at the floor) and it reproduced on the test set: out-of-fold precision 0.400, test precision 0.407. I would rather publish that than a heroic threshold that collapses.

The calibrator is fit on out-of-fold scores from the fold models, then applied to a refit model. The refit is trained on more data, so its raw scores are not identical to the out-of-fold scores. A stricter design holds out a calibration slice from the training set and never uses those rows to fit the final model. I used the out-of-fold route so every training row still influences the model. The test calibration numbers are the check that this was good enough.

Serving from a joblib file means a deploy is "copy a file," not "hope the registry is reachable." The cost is two sources of truth. The registry records the version. The file is what production runs. `model_version` is stamped into both.

The drift control is simulated. It proves the math fires on a shift and stays quiet on a matched draw. It does not prove we would notice a subtle fraud-mix change in production. The cutoffs 0.10 and 0.20 are conventions. They live in config so they can be changed without a code edit.

## How to run and demo it

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make train
make test
uvicorn payguard.serving.app:app --port 8000
```

Then:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/model
curl -s http://127.0.0.1:8000/predict -H 'content-type: application/json' -d @reports/sample_request.json
curl -s http://127.0.0.1:8000/explain -H 'content-type: application/json' -d @reports/sample_request.json
```

`/docs` is the generated OpenAPI page. For the high-risk contrast, post a transaction with `channel=online`, `country_match=foreign`, `device_trust=new`, `hour=2`, `merchant_category=gambling`, and an amount several times `avg_amount_30d`. On the committed model that scores about 0.9997 and flags.

`docker compose up --build` starts MLflow on port 5000 and the API on port 8000, with `./models` mounted into the API container. The image was not built in the environment that produced these metrics; the API check above was uvicorn against the joblib bundle.

`make ci` trains the small config into `models/ci/` and `reports/ci/` so it does not overwrite this report. GitHub Actions runs ruff, pytest, and that same command.

Rebuild the PDF from this file with `make pdf`.

## Limitations and what I would do next

The labels are generated. A model can look sharp because it is recovering `fraud_effects`, not because it would survive a real adversary. I would not quote 0.283 PR-AUC as a claim about any bank's book.

There is no concept drift in the training rows themselves. The shifted batch is a scripted stress test. Production drift is slower and mixed with seasonality, product launches, and a change in who gets labelled.

Labels would be delayed. Training on "fraud as known today" undercounts recent fraud. I would train on a mature cohort and evaluate alerts at the time they would have been worked.

The precision floor is not a dollar loss. If a missed fraud costs fifty times a wasted review, the right threshold is lower than 0.40 and the queue is larger. I would put that cost in the config and choose the threshold that minimises expected cost on out-of-fold scores.

The retrain flag does not train. The next step is a job that, given fresh labels, runs this same pipeline and registers a new version only if out-of-fold PR-AUC improves. The previous version stays up until then.

SHAP is global on a sample and local for one row. It is not a full dependence plot, and the linear explainer attributes one-hot pieces separately (`country_match_foreign` and `country_match_domestic` both appear). I would group those for a reviewer-facing screen.

## Interview questions

### 1. Give me the 60-second version.

Card fraud is rare, so accuracy is a trap. PayGuard builds a checked transaction table, compares logistic regression with two boosting models under stratified cross-validation, and ships the model with the best out-of-fold PR-AUC. On this run that is logistic regression: test PR-AUC 0.283 versus a 0.0215 baseline, test ROC-AUC 0.870. A threshold frozen on out-of-fold scores holds test precision at 0.407 while flagging 0.9% of transactions and catching 17% of the fraud. Platt scaling cuts expected calibration error from 0.315 to 0.007. FastAPI serves that bundle. MLflow records it. PSI on the score is 0.004 on a matched batch and 6.13 on a shifted one, and only the shifted batch raises the retrain flag.

### 2. Why is accuracy the wrong metric?

Because the negative class is 97.86% of the table. Predicting "legitimate" for every row scores about that, with zero recall. I need a metric that ignores the mountain of easy negatives and asks whether the rows I rank at the top are actually fraud. That is PR-AUC. I still report the confusion counts at the operating point, because a single area cannot tell a reviewer how many alerts they will work tomorrow.

### 3. Why select on PR-AUC rather than ROC-AUC?

ROC-AUC compares a random fraud with a random legitimate transaction, across the whole score range. Most of that range is never used. We only act on the top of the ranking. PR-AUC is the area of precision against recall, so it penalises a model whose top scores are littered with false positives. On this run all three models have test ROC-AUC above 0.82, while test PR-AUC ranges from 0.229 to 0.283. The gap between those PR-AUC numbers is the decision. ROC-AUC stays in the report so I can answer this question with both.

### 4. Walk through the operating point with the actual counts.

The floor is precision 0.40. On calibrated out-of-fold scores the highest recall at that floor is 0.167, at threshold 0.256, from 86 true positives and 129 false positives. I freeze 0.256. On the 6,000 test rows the refit model flags 54 transactions, 0.9%. Twenty-two are fraud, 32 are not, so precision is 22/54 = 0.407. Fraud in the test set is 129 cases (22 + 107). Recall is 22/129 = 0.171. The floor held. We still miss 107 frauds. I would say that out loud rather than round it into "the model detects fraud."

### 5. Why did logistic regression beat XGBoost?

Because the feature pipeline already contains the interactions the generator uses: foreign-and-online, new-device-at-night, amount versus trailing average, night, velocity. A linear model with those columns can represent the process. Boosting did not find extra ranking signal; its out-of-fold PR-AUC is 0.160 for XGBoost and 0.156 for histogram gradient boosting, against 0.223 for logistic regression. The fold standard deviations (0.019, 0.027, 0.030) do not overlap enough to call this a tie. I would still keep the tree models in the bake-off. The day the data has interactions I did not code, they get a chance to win without a redesign.

### 6. Why not SMOTE or random undersampling?

Three reasons. Fitting SMOTE before the split lets synthetic neighbours carry information across the fold boundary. Even inside the fold, synthetic points change the prevalence, so the probabilities are no longer probabilities of the real stream. And the thing I actually want to control is the alert rate, which is a threshold on a well-ranked score, not a change to the training distribution. Class weights (`class_weight="balanced"`, and `scale_pos_weight` for XGBoost) up-weight the rare class without inventing rows. The calibration step then puts the probabilities back on the real prevalence. The Brier move from 0.146 to 0.018 is that correction.

### 7. Where does leakage sneak in, and how is it blocked?

The usual holes are: scaling with the test mean, choosing one-hot categories on the full frame, tuning the threshold on the test set, and resampling before the split. `evaluate_candidate` fits `make_preprocessor` on the fold's training rows only, then transforms the validation rows. The final preprocessor is fit on the training frame only. The threshold is chosen on out-of-fold training scores. The test set is transformed and scored after that. The feature builder cannot leak a global statistic because every new column is a function of that row. Target encoding would be a different story; I did not use it.

### 8. What did calibration actually change?

Not the ranking. Platt scaling is monotone, so PR-AUC and ROC-AUC are properties of the raw score. It changed the meaning of a number like 0.25. Before calibration, balanced logistic regression was shouting: test Brier 0.146, ECE 0.315, and the reliability curve is far below the diagonal. After, Brier is 0.018 and ECE is 0.007. The threshold of 0.256 is a calibrated probability. Using the same numeric cutoff on the raw score would flag a completely different set of people. I threshold only the calibrated output, in `predict_calibrated`.

### 9. Why fit the calibrator on out-of-fold scores?

If I fit it on in-sample scores, the model is over-confident on rows it just trained on, and the calibrator learns that over-confidence. Out-of-fold scores are predictions on rows the fold model did not train on, which is the right input for a map we will apply to future rows. I then apply that map to the refit model. Those two models are not identical. The test ECE of 0.007 is the evidence the transfer worked here. If it had not, I would hold out a calibration slice and accept a slightly smaller training set.

### 10. Walk through one scoring request.

FastAPI parses the JSON into `Transaction`. A negative amount or a category outside the literal sets never reaches the model; that is 422. The row becomes a one-row frame in the column order of `RAW_FEATURE_COLUMNS`. `predict_calibrated` runs `TransactionFeatureBuilder`, then the imputer, scaler, and one-hot encoder that were fit on the training frame, then `predict_proba`, then the Platt model. The flag is `probability >= threshold`. The response includes the threshold and `model_version` so a reviewer can see which cutoff and which file produced the bit. `/explain` does the same score, then SHAP on the encoded row.

### 11. What is PSI, using your two batches?

PSI asks whether a new sample has the same shape as the reference, one feature or one score at a time. I cut the reference into quantile bins and compare the share of rows in each bin. Under about 0.10 is the usual "stable" band, 0.10 to 0.20 is a watch, above 0.20 is a shift. Those are heuristics; they are config values, not theorems. On a new draw from the same generator, prediction PSI is 0.004. On the scripted shift, it is 6.135. The matched draw does not retrain. The shifted draw does, both because the score moved and because eight input features crossed 0.20. KS on the score agrees: 0.014 versus 0.814.

### 12. How is a schema failure different from drift?

A schema failure means the row is not a legal transaction: negative amount, hour 27, a category we have never defined. We refuse to score it. Drift means the rows are legal and the population has moved: more foreign online volume, larger amounts, a different hour mix. We can still score them. The monitor says whether the score distribution, or enough inputs, moved far enough that the old model should not be trusted without a new fit. Both can be true on the same day. They are different code paths: `validate_transactions` versus `build_drift_report`.

### 13. The retrain flag fired. What happens next?

In this repository, a markdown and JSON report. I do not automatically fit a new model on the drifted batch. Those rows may not have labels yet, and training on old labels plus new features teaches the wrong conditional. The production version of this flag opens a ticket: pull a mature labelled window, rerun `run_pipeline`, and register a new version only if out-of-fold PR-AUC beats the champion. Until that happens the old bundle keeps serving. A silent retrain is worse than a stale model you know about.

### 14. A customer asks why they were declined. What do you show?

I would not show a raw SHAP number as "this feature added 12% to your fraud probability." The values are in the model's margin, the log-odds space, and the note on `/explain` says that. I would show the direction and the rank: amount, online channel, a new device, a foreign country pushed the score up, and I would show the calibrated probability next to the threshold. For the high-risk demo row the probability is 0.9997 against a threshold of 0.256, so the decision is not a close call. For a row near 0.26 I would say it is near the cutoff and a reviewer should look, not that a single feature "caused" the decline. Global SHAP is dominated by `log_amount` and `channel_online`, which matches the generator, and I would say that is a sanity check on the model, not a customer-facing sentence.

### 15. Why is the training data synthetic? What will you not claim?

The public card-fraud file most people know is anonymised principal components and is awkward to redistribute. Synthetic payments with named fields make the schema, the features, and the SHAP story checkable, and they let CI run offline. The generator's fraud rate, the interactions, and the label noise are in `generate.py`, so I can point at them. I will not claim the 0.283 test PR-AUC is what this model would score on a bank's production stream. A real book has adversaries, delayed labels, and features I did not think of. The engineering around the model is what transfers. The metric is a property of this generator and this seed.

### 16. How is the model versioned, and which bytes does production run?

`run_pipeline` stamps `model_version` as the model name plus a UTC timestamp, here `logistic_regression-20261120071250`. That string is inside the joblib bundle and in every API response. MLflow has a separate id: run `49fe7559f07b43e9abd657a0a85a3f5e`, logged-model URI `models:/m-9662271936424ee5b3dc1e265dfd0d71`, registry version 1 of `payguard-fraud-detector`. Production runs the joblib file, mounted into the API container. The registry is how I answer "what did we train, with which metrics?" six months later. If those two diverge, the API's `/model` payload is the one to believe for a live decision, and the mismatch is an incident.

### 17. Why not call MLflow on the request path?

Latency, and a boring failure mode. A scoring request should depend on a local file and a process, not on a tracking server. If sqlite or the MLflow container is down, `/health` should still be ok and `/predict` should still return. The pyfunc wrapper exists so the registry entry is actually loadable later. It is not in the hot path. `create_app` only `joblib.load`s the bundle.

### 18. How would you set the threshold from money instead of from a precision floor?

Assign a cost to a false positive (a review, or a wrongly declined sale) and a cost to a false negative (loss given fraud, times the chance we never recover it). On out-of-fold scores, sweep the threshold and pick the one with the lowest total cost. Precision 0.40 is a stand-in for "a reviewer is not wasting most of their hour." It is not a loss function. If false negatives are very expensive I would accept precision of 0.15 and a larger queue, and I would show the cost curve next to the precision-recall curve. I would still choose the threshold on out-of-fold scores, not on the test set.

### 19. What does the CI job actually prove?

`.github/workflows/ci.yml` installs the package on Python 3.11, runs ruff, runs pytest, and runs `configs/ci.yaml`: 2,500 rows, 3 folds, small trees, synthetic data, its own output directory. Pytest covers the schema, the feature flags, the precision-floor helper, PSI on a known shift, a tiny three-model training that beats the prevalence baseline, MLflow registration, and the API (health, 422s, batch ordering, explain). The CI training job proves the pipeline entry point still runs, not that the committed 30,000-row metrics were recomputed. Those metrics are an artifact of the full config. If someone changes the generator, both the tests and a new full run have to move together.

### 20. What breaks first if this is pointed at a real stream?

Label delay, then feature availability at decision time, then adversaries. The generator's features are all known at the moment of the transaction. A real `avg_amount_30d` has to be computed from history that exists at authorisation, not from a warehouse join that includes the current row. Fraudsters will move toward whatever slice the model is not flagging; a fixed generator cannot show that. The third break is calibration under a new mix: a model can keep ranking while its probabilities drift, which is why I watch prediction PSI and would add expected calibration error on a live window.

### 21. PSI is quiet but the model feels worse. What else do you watch?

PSI can miss a shift that stays inside the same bins but changes the label rate, and it can miss a change in the relationship between features and fraud. I would monitor, on a labelled mature window, the precision and recall at the frozen threshold, PR-AUC, and ECE. A drop in precision at the same threshold with a stable PSI means the conditional `p(fraud | x)` moved even though `p(x)` did not. That is concept drift, and it is a retrain even when the feature PSI report is green. The simulated shift in this repo is covariate shift, which PSI is good at. I would not pretend the same report covers concept drift.

### 22. Boosting wins on the test set and loses on out-of-fold PR-AUC. Which do you ship?

The out-of-fold winner. The test set is one split. Using it to overturn the selection rule is how you overfit the confirmation set and then have nothing left to report. On this run I did not have to make that call: logistic regression leads on both. If it had not, I would ship the out-of-fold winner, put the disagreement in the report, and, if the gap mattered, rerun with more folds or a second seed before touching the test set again.

### 23. Why standardise features for a tree model?

One preprocessor serves all three candidates, and logistic regression needs the scale. Trees do not, but monotone scaling does not change their splits in any way I care about here. Sharing the preprocessor means the serving bundle has one code path. The cost is a few extra columns of standardised binaries such as `is_night`. That is cheap next to maintaining two feature pipelines.

### 24. What would you ask the reviewer queue for, after the first month?

The labels on the alerts we showed them, including the ones they marked legitimate, and a sample of the transactions we did not show them. Without the second sample I can estimate precision and I cannot estimate recall. I would also ask how often a "fraud" label is reversed later. If reversals are common, the 0.407 test precision is an upper bound on precision against the final label, and I would train on the matured label, not the first one.
