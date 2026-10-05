# Weekly drift report

Alert rule: PSI > 0.2 on any monitored feature or on the model score.
Reference window: validation week 9.

| Week | Alert | Max PSI | Feature at max |
| --- | --- | --- | --- |
| 10 | no | 0.0064 | cust_prior_mean_amount |
| 11 | yes | 20.9958 | acq_region |
| 12 | no | 0.0474 | cust_prior_mean_amount |

Week 11 changes the fraud-type mix (more card testing and device-farm fraud) and routes every transaction to acquirer region 0. Chi-square for the week 10 vs week 11 fraud-type counts is 700.0000.

Stable weeks 10 and 12 keep the training fraud mix and the balanced region mix.

Fraud type codes: 0=legit, 1=card_test, 2=ato, 3=mule, 4=device_farm.
