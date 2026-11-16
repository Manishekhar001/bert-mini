## Test-set results (raw model output)

Always guessing the most common class scores 0.342 accuracy on this test set. Validation and test keep the natural class mix; training is rebalanced, so the raw numbers below are not the best these models can do (see the next table).

| model | parameters (M) | accuracy | macro-F1 | ECE |
| --- | ---: | ---: | ---: | ---: |
| tfidf+logreg |  | 0.600 | 0.313 | 0.106 |
| bert-uncased-ft | 11.2 | 0.597 | 0.295 | 0.039 |
| D_full_ft | 11.2 | 0.591 | 0.288 | 0.044 |
| D2_full_ft_epoch2 | 11.2 | 0.566 | 0.279 | 0.056 |
| C2_epoch2 | 11.2 | 0.556 | 0.272 | 0.035 |
| C_distill_domain | 11.2 | 0.555 | 0.264 | 0.033 |

## After per-class prior correction and temperature scaling (fitted on validation)

| model | correction strength | accuracy | macro-F1 | ECE |
| --- | ---: | ---: | ---: | ---: |
| D_full_ft | 1.00 | 0.638 | 0.299 | 0.014 |
| D2_full_ft_epoch2 | 1.00 | 0.629 | 0.295 | 0.027 |
| bert-uncased-ft | 1.00 | 0.638 | 0.294 | 0.017 |
| C_distill_domain | 1.00 | 0.604 | 0.268 | 0.021 |
| C2_epoch2 | 1.00 | 0.603 | 0.266 | 0.025 |

Strength 1.00 applies the whole fitted correction and 0.00 none; it is chosen by cross-validation inside the validation set. The blend row mixes two models' log-probabilities and is not part of the packaged model.

## Per-class test results: packaged model (C_distill_domain)

| class | test reports | precision | recall | F1 | most often mistaken for |
| --- | ---: | ---: | ---: | ---: | --- |
| humanfactors | 3371 | 0.63 | 0.56 | 0.59 | procedure (362) |
| aircraft | 3309 | 0.80 | 0.80 | 0.80 | humanfactors (261) |
| procedure | 967 | 0.24 | 0.20 | 0.22 | humanfactors (385) |
| ambiguous | 796 | 0.34 | 0.18 | 0.23 | humanfactors (228) |
| weather | 329 | 0.43 | 0.75 | 0.55 | humanfactors (33) |
| airport | 244 | 0.42 | 0.35 | 0.38 | humanfactors (65) |
| environment-nonweatherrelated | 238 | 0.19 | 0.20 | 0.20 | humanfactors (60) |
| chartorpublication | 134 | 0.40 | 0.55 | 0.46 | procedure (16) |
| airspacestructure | 129 | 0.18 | 0.41 | 0.25 | humanfactors (42) |
| atcequipment/navfacility/buildings | 126 | 0.21 | 0.33 | 0.26 | environment-nonweatherrelated (22) |
| softwareandautomation | 68 | 0.09 | 0.04 | 0.06 | humanfactors (13) |
| companypolicy | 53 | 0.07 | 0.28 | 0.11 | humanfactors (8) |
| equipment/tooling | 38 | 0.09 | 0.26 | 0.13 | humanfactors (11) |
| mel | 24 | 0.13 | 0.46 | 0.20 | aircraft (8) |
| incorrect/notinstalled/unavailablepart | 19 | 0.07 | 0.21 | 0.11 | aircraft (5) |
| staffing | 11 | 0.08 | 0.36 | 0.13 | procedure (3) |
| manuals | 10 | 0.06 | 0.10 | 0.07 | procedure (4) |
| logbookentry | 2 | 0.00 | 0.00 | 0.00 | humanfactors (1) |

## Confusion matrix: packaged model

Each row is a true class; each cell is the percent of that class predicted as the column's class (bold = correct).

| true class \ predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 aircraft (n=3309) | **80** |  |  | 2 | 1 |  | 1 | 1 | 1 | 8 |  |  |  | 1 | 2 |  |  | 2 |
| 1 airport (n=244) | 6 | **35** | 1 | 2 | 2 | 7 | 2 | 3 |  | 27 |  |  |  |  | 9 |  |  | 5 |
| 2 airspacestructure (n=129) | 1 | 1 | **41** | 2 | 2 | 3 |  | 3 |  | 33 |  |  |  |  | 13 | 1 |  | 1 |
| 3 ambiguous (n=796) | 12 | 2 | 4 | **18** | 3 | 2 | 3 | 6 | 1 | 29 | 1 |  |  |  | 11 | 1 |  | 8 |
| 4 atcequipment/navfacility/buildings (n=126) | 3 | 4 | 3 | 13 | **33** | 5 | 5 | 17 |  | 6 |  |  |  |  | 3 | 2 | 4 | 2 |
| 5 chartorpublication (n=134) | 3 | 10 | 4 | 1 | 1 | **55** |  | 1 |  | 5 | 1 |  | 1 |  | 12 | 3 |  | 2 |
| 6 companypolicy (n=53) | 8 |  | 4 | 2 |  | 6 | **28** | 2 | 2 | 15 |  |  |  | 2 | 13 |  | 11 | 8 |
| 7 environment-nonweatherrelated (n=238) | 3 | 3 | 3 | 16 | 7 | 1 | 5 | **20** |  | 25 |  |  |  |  | 8 |  | 4 | 5 |
| 8 equipment/tooling (n=38) | 13 |  |  | 3 | 8 |  | 3 | 3 | **26** | 29 | 8 |  |  |  | 8 |  |  |  |
| 9 humanfactors (n=3371) | 11 | 2 | 3 | 4 | 2 | 1 | 2 | 1 | 1 | **56** | 1 |  |  | 1 | 11 |  |  | 4 |
| 10 incorrect/notinstalled/unavailablepart (n=19) | 26 |  |  |  |  |  | 11 |  |  | 21 | **21** | 5 | 5 |  | 11 |  |  |  |
| 11 logbookentry (n=2) |  |  |  |  |  |  |  |  |  | 50 |  |  |  | 50 |  |  |  |  |
| 12 manuals (n=10) | 20 |  |  |  |  |  | 20 |  |  |  | 10 |  | **10** |  | 40 |  |  |  |
| 13 mel (n=24) | 33 |  |  |  |  |  | 4 |  |  |  |  | 4 |  | **46** | 12 |  |  |  |
| 14 procedure (n=967) | 11 | 1 | 6 | 3 | 2 | 2 | 3 | 2 | 2 | 40 | 1 | 1 |  | 2 | **20** |  | 1 | 2 |
| 15 softwareandautomation (n=68) | 13 |  | 6 | 12 | 10 | 16 |  | 4 |  | 19 | 3 |  |  |  | 10 | **4** |  | 1 |
| 16 staffing (n=11) |  |  |  |  |  |  | 18 |  |  | 18 |  |  |  |  | 27 |  | **36** |  |
| 17 weather (n=329) | 8 | 2 | 2 | 1 |  |  |  | 2 |  | 10 |  |  |  |  | 1 |  | 1 | **75** |

Column key: 0 = aircraft; 1 = airport; 2 = airspacestructure; 3 = ambiguous; 4 = atcequipment/navfacility/buildings; 5 = chartorpublication; 6 = companypolicy; 7 = environment-nonweatherrelated; 8 = equipment/tooling; 9 = humanfactors; 10 = incorrect/notinstalled/unavailablepart; 11 = logbookentry; 12 = manuals; 13 = mel; 14 = procedure; 15 = softwareandautomation; 16 = staffing; 17 = weather.

## CPU latency (4 threads)

| backend | size (MB) | batch-1 median (ms) | batch-1 p95 (ms) | batch-32 (ms/report) |
| --- | ---: | ---: | ---: | ---: |
| torch | 44.7 | 15.7 | 32.5 | 28.75 |
| onnx | 44.8 | 11.4 | 21.9 | 20.72 |
| onnx_int8 | 11.4 | 7.7 | 17.8 | 12.00 |
