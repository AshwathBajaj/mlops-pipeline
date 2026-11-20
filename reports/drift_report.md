# Drift report

PSI uses reference quantiles. KS is reported for numeric features and for the score.
The retrain flag is the rule in `configs/train.yaml`, not a human review.

## stable_draw

- Reference rows: 4000
- Current rows: 4000
- Prediction PSI: 0.0040
- Prediction KS: 0.0140
- Retrain: false
- Reason: none

| Feature | Kind | PSI | KS |
| --- | --- | ---: | ---: |
| amount | numeric | 0.0022 | 0.0138 |
| hour | numeric | 0.0019 | 0.0127 |
| account_age_days | numeric | 0.0050 | 0.0202 |
| credit_limit | numeric | 0.0023 | 0.0190 |
| utilization | numeric | 0.0054 | 0.0227 |
| txn_count_24h | numeric | 0.0011 | 0.0100 |
| avg_amount_30d | numeric | 0.0028 | 0.0182 |
| days_since_last_txn | numeric | 0.0023 | 0.0105 |
| distance_from_home_km | numeric | 0.0041 | 0.0195 |
| merchant_risk_score | numeric | 0.0023 | 0.0120 |
| failed_attempts_24h | numeric | 0.0004 | 0.0075 |
| merchant_category | categorical | 0.0039 |  |
| channel | categorical | 0.0009 |  |
| country_match | categorical | 0.0005 |  |
| device_trust | categorical | 0.0009 |  |

## shifted_draw

- Reference rows: 4000
- Current rows: 4000
- Prediction PSI: 6.1348
- Prediction KS: 0.8138
- Retrain: true
- Reason: 8 features have PSI >= 0.2: amount, hour, txn_count_24h, distance_from_home_km, merchant_risk_score, channel, country_match, device_trust
- Reason: prediction PSI 6.1348 >= 0.1

| Feature | Kind | PSI | KS |
| --- | --- | ---: | ---: |
| amount | numeric | 1.0913 | 0.4178 |
| hour | numeric | 1.8091 | 0.4048 |
| account_age_days | numeric | 0.0050 | 0.0202 |
| credit_limit | numeric | 0.0023 | 0.0190 |
| utilization | numeric | 0.0054 | 0.0227 |
| txn_count_24h | numeric | 13.0300 | 0.9660 |
| avg_amount_30d | numeric | 0.0028 | 0.0182 |
| days_since_last_txn | numeric | 0.0023 | 0.0105 |
| distance_from_home_km | numeric | 12.4338 | 0.9520 |
| merchant_risk_score | numeric | 11.0091 | 0.8688 |
| failed_attempts_24h | numeric | 0.0004 | 0.0075 |
| merchant_category | categorical | 0.0039 |  |
| channel | categorical | 0.3893 |  |
| country_match | categorical | 0.6016 |  |
| device_trust | categorical | 0.3776 |  |
