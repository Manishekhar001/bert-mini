## Test-set results (raw model output)

Always guessing the most common class scores 0.342 accuracy on this test set. Validation and test keep the natural class mix; training is rebalanced, so the raw numbers below are not the best these models can do (see the next table).

| model | parameters (M) | accuracy | macro-F1 | ECE |
| --- | ---: | ---: | ---: | ---: |
| tfidf+logreg |  | 0.600 | 0.313 | 0.106 |
| D_full_ft | 11.2 | 0.591 | 0.288 | 0.044 |
| D2_full_ft_epoch2 | 11.2 | 0.566 | 0.279 | 0.056 |
| C2_epoch2 | 11.2 | 0.556 | 0.272 | 0.035 |
| C_distill_domain | 11.2 | 0.555 | 0.264 | 0.033 |

## After per-class prior correction and temperature scaling (fitted on validation)

| model | correction strength | accuracy | macro-F1 | ECE |
| --- | ---: | ---: | ---: | ---: |
| D_full_ft | 1.00 | 0.638 | 0.299 | 0.014 |
| D2_full_ft_epoch2 | 1.00 | 0.629 | 0.295 | 0.027 |
| C_distill_domain | 1.00 | 0.604 | 0.268 | 0.021 |
| C2_epoch2 | 1.00 | 0.603 | 0.266 | 0.025 |

Strength 1.00 applies the whole fitted correction and 0.00 none; it is chosen by cross-validation inside the validation set. The blend row mixes two models' log-probabilities and is not part of the packaged model.

## Per-class test results: packaged model (D_full_ft)

| class | test reports | precision | recall | F1 | most often mistaken for |
| --- | ---: | ---: | ---: | ---: | --- |
| humanfactors | 3371 | 0.64 | 0.76 | 0.69 | aircraft (307) |
| aircraft | 3309 | 0.81 | 0.86 | 0.84 | humanfactors (235) |
| procedure | 967 | 0.32 | 0.15 | 0.20 | humanfactors (536) |
| ambiguous | 796 | 0.41 | 0.24 | 0.30 | humanfactors (331) |
| weather | 329 | 0.57 | 0.67 | 0.62 | aircraft (45) |
| airport | 244 | 0.52 | 0.45 | 0.48 | humanfactors (78) |
| environment-nonweatherrelated | 238 | 0.27 | 0.10 | 0.14 | humanfactors (90) |
| chartorpublication | 134 | 0.43 | 0.63 | 0.51 | procedure (15) |
| airspacestructure | 129 | 0.43 | 0.22 | 0.30 | humanfactors (70) |
| atcequipment/navfacility/buildings | 126 | 0.42 | 0.39 | 0.40 | ambiguous (25) |
| softwareandautomation | 68 | 0.00 | 0.00 | 0.00 | humanfactors (18) |
| companypolicy | 53 | 0.09 | 0.15 | 0.11 | humanfactors (16) |
| equipment/tooling | 38 | 0.11 | 0.16 | 0.13 | humanfactors (19) |
| mel | 24 | 0.18 | 0.42 | 0.25 | aircraft (8) |
| incorrect/notinstalled/unavailablepart | 19 | 0.09 | 0.32 | 0.13 | humanfactors (6) |
| staffing | 11 | 0.09 | 0.45 | 0.15 | humanfactors (4) |
| manuals | 10 | 0.17 | 0.10 | 0.12 | humanfactors (3) |
| logbookentry | 2 | 0.00 | 0.00 | 0.00 | humanfactors (2) |

## Confusion matrix: packaged model

Each row is a true class; each cell is the percent of that class predicted as the column's class (bold = correct).

| true class \ predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 aircraft (n=3309) | **86** |  |  | 2 |  |  |  | 1 |  | 7 | 1 |  |  |  | 1 |  |  | 1 |
| 1 airport (n=244) | 6 | **45** |  | 1 | 2 | 3 | 1 | 1 | 1 | 32 |  |  |  |  | 2 |  |  | 4 |
| 2 airspacestructure (n=129) | 2 | 1 | **22** | 2 | 1 | 2 |  |  |  | 54 |  |  |  |  | 12 |  | 2 | 1 |
| 3 ambiguous (n=796) | 14 | 3 | 1 | **24** | 1 | 2 | 2 | 2 | 1 | 42 |  |  |  |  | 5 |  |  | 4 |
| 4 atcequipment/navfacility/buildings (n=126) | 7 | 7 | 2 | 20 | **39** | 5 | 1 | 10 | 1 | 6 |  |  |  |  | 2 |  | 1 | 1 |
| 5 chartorpublication (n=134) | 2 | 10 | 1 | 1 | 1 | **63** |  |  |  | 10 |  |  |  |  | 11 |  |  | 1 |
| 6 companypolicy (n=53) | 8 |  |  | 2 |  | 4 | **15** | 2 | 2 | 30 | 2 |  | 2 | 2 | 13 |  | 13 | 6 |
| 7 environment-nonweatherrelated (n=238) | 7 | 4 |  | 22 | 2 | 1 | 3 | **10** | 1 | 38 |  |  |  |  | 6 |  | 4 | 1 |
| 8 equipment/tooling (n=38) | 13 | 8 |  | 3 | 3 |  |  |  | **16** | 50 | 3 |  | 3 |  | 3 |  |  |  |
| 9 humanfactors (n=3371) | 9 | 1 |  | 2 |  | 1 | 1 |  |  | **76** | 1 |  |  |  | 5 |  |  | 2 |
| 10 incorrect/notinstalled/unavailablepart (n=19) | 21 |  |  |  |  |  | 5 |  |  | 32 | **32** | 5 |  | 5 |  |  |  |  |
| 11 logbookentry (n=2) |  |  |  |  |  |  |  |  |  | 100 |  |  |  |  |  |  |  |  |
| 12 manuals (n=10) | 10 |  |  |  |  |  | 10 |  |  | 30 | 20 |  | **10** |  | 20 |  |  |  |
| 13 mel (n=24) | 33 |  |  |  |  |  |  |  |  | 21 |  | 4 |  | **42** |  |  |  |  |
| 14 procedure (n=967) | 11 | 1 | 2 | 4 | 1 | 2 | 1 |  | 1 | 55 | 2 |  |  | 1 | **15** |  | 1 | 2 |
| 15 softwareandautomation (n=68) | 15 |  |  | 21 | 1 | 24 | 3 | 3 |  | 26 | 1 | 1 |  |  | 1 |  |  | 3 |
| 16 staffing (n=11) |  |  |  |  |  |  |  | 9 |  | 36 |  |  |  |  | 9 |  | **45** |  |
| 17 weather (n=329) | 14 | 2 |  | 3 |  |  |  |  |  | 13 |  |  |  |  | 1 |  | 1 | **67** |

Column key: 0 = aircraft; 1 = airport; 2 = airspacestructure; 3 = ambiguous; 4 = atcequipment/navfacility/buildings; 5 = chartorpublication; 6 = companypolicy; 7 = environment-nonweatherrelated; 8 = equipment/tooling; 9 = humanfactors; 10 = incorrect/notinstalled/unavailablepart; 11 = logbookentry; 12 = manuals; 13 = mel; 14 = procedure; 15 = softwareandautomation; 16 = staffing; 17 = weather.

## CPU latency (4 threads)

| backend | size (MB) | batch-1 median (ms) | batch-1 p95 (ms) | batch-32 (ms/report) |
| --- | ---: | ---: | ---: | ---: |
| torch | 44.7 | 14.9 | 29.8 | 20.72 |
