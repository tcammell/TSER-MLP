# TSER-MLP

MLP-only models for time series extrinsic regression using TSMixer [1] and gMLP [2], 
on the datasets described in [3].

Trainer cli will automatically download datasets from http://tseregression.org/

Train and test:

`python trainer.py -c config/<xxx>.yaml`

Optuna tuning:

`python tuner.py -d <dataset> -n <trials>`

Results directory contains critical difference diagrams versus results from [4] and [5].


References:

[1] Si-An Chen et al. “TSMixer: An All-MLP Architecture for Time Series Forecasting”. 
In: Transactions on Machine Learning Research (Apr. 24, 2023). issn: 2835-8856.

[2] Hanxiao Liu et al. “Pay Attention to MLPs”. In: Advances in Neural
Information Processing Systems. Vol. 34. Curran Associates, Inc., 2021,
pp. 9204–9215.

[3] Chang Wei Tan et al. Monash University, UEA, UCR Time Series Extrinsic Regression 
Archive. Oct. 19, 2020. arXiv: 2006.10996[cs,stat].

[4] Chang Wei Tan et al. “Time series extrinsic regression”. In: Data Mining
and Knowledge Discovery 35.3 (May 1, 2021), pp. 1032–1060. issn: 1573-
756X. doi: 10.1007/s10618-021-00745-9.

[5] David Guijo-Rubio et al. Unsupervised Feature Based Algorithms for Time Series 
Extrinsic Regression. May 2, 2023. doi: 10.48550/arXiv.2305.01429. arXiv: 2305.01429[cs,stat].


