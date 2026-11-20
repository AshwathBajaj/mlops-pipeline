# Drift report

PSI uses reference quantiles. KS is reported for numeric features and for the score.
The retrain flag is the rule in `configs/train.yaml`, not a human review.

## stable_draw

- Reference rows: 4000
- Current rows: 4000
- Prediction PSI: 0.0066
- Prediction KS: 0.0145
- Retrain: false
- Reason: none

| Feature | Kind | PSI | KS |
| --- | --- | ---: | ---: |
| Time | numeric | 0.0065 | 0.0243 |
| Amount | numeric | 0.0032 | 0.0192 |
| V1 | numeric | 0.0055 | 0.0150 |
| V2 | numeric | 0.0030 | 0.0135 |
| V3 | numeric | 0.0048 | 0.0190 |
| V4 | numeric | 0.0062 | 0.0185 |
| V5 | numeric | 0.0039 | 0.0198 |
| V6 | numeric | 0.0027 | 0.0223 |
| V7 | numeric | 0.0041 | 0.0280 |
| V8 | numeric | 0.0076 | 0.0165 |
| V9 | numeric | 0.0013 | 0.0105 |
| V10 | numeric | 0.0046 | 0.0170 |
| V11 | numeric | 0.0054 | 0.0195 |
| V12 | numeric | 0.0073 | 0.0278 |
| V13 | numeric | 0.0034 | 0.0160 |
| V14 | numeric | 0.0086 | 0.0142 |
| V15 | numeric | 0.0080 | 0.0305 |
| V16 | numeric | 0.0065 | 0.0250 |
| V17 | numeric | 0.0033 | 0.0170 |
| V18 | numeric | 0.0050 | 0.0132 |
| V19 | numeric | 0.0061 | 0.0275 |
| V20 | numeric | 0.0018 | 0.0115 |
| V21 | numeric | 0.0057 | 0.0180 |
| V22 | numeric | 0.0009 | 0.0078 |
| V23 | numeric | 0.0048 | 0.0150 |
| V24 | numeric | 0.0053 | 0.0168 |
| V25 | numeric | 0.0080 | 0.0198 |
| V26 | numeric | 0.0045 | 0.0180 |
| V27 | numeric | 0.0100 | 0.0228 |
| V28 | numeric | 0.0031 | 0.0153 |

## shifted_draw

- Reference rows: 4000
- Current rows: 4000
- Prediction PSI: 3.8083
- Prediction KS: 0.6520
- Retrain: true
- Reason: 7 features have PSI >= 0.2: Time, Amount, V1, V2, V3, V4, V14
- Reason: prediction PSI 3.8083 >= 0.1

| Feature | Kind | PSI | KS |
| --- | --- | ---: | ---: |
| Time | numeric | 1.9849 | 0.3445 |
| Amount | numeric | 0.3094 | 0.2022 |
| V1 | numeric | 0.7936 | 0.5610 |
| V2 | numeric | 2.7282 | 0.6715 |
| V3 | numeric | 1.3960 | 0.5040 |
| V4 | numeric | 4.9364 | 0.7830 |
| V5 | numeric | 0.0039 | 0.0198 |
| V6 | numeric | 0.0027 | 0.0223 |
| V7 | numeric | 0.0041 | 0.0280 |
| V8 | numeric | 0.0076 | 0.0165 |
| V9 | numeric | 0.0013 | 0.0105 |
| V10 | numeric | 0.0046 | 0.0170 |
| V11 | numeric | 0.0054 | 0.0195 |
| V12 | numeric | 0.0073 | 0.0278 |
| V13 | numeric | 0.0034 | 0.0160 |
| V14 | numeric | 11.0844 | 0.9655 |
| V15 | numeric | 0.0080 | 0.0305 |
| V16 | numeric | 0.0065 | 0.0250 |
| V17 | numeric | 0.0033 | 0.0170 |
| V18 | numeric | 0.0050 | 0.0132 |
| V19 | numeric | 0.0061 | 0.0275 |
| V20 | numeric | 0.0018 | 0.0115 |
| V21 | numeric | 0.0057 | 0.0180 |
| V22 | numeric | 0.0009 | 0.0078 |
| V23 | numeric | 0.0048 | 0.0150 |
| V24 | numeric | 0.0053 | 0.0168 |
| V25 | numeric | 0.0080 | 0.0198 |
| V26 | numeric | 0.0045 | 0.0180 |
| V27 | numeric | 0.0100 | 0.0228 |
| V28 | numeric | 0.0031 | 0.0153 |
