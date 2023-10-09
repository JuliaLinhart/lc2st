# lc2st: local classifier two-sample tests
Official code for [L-C2ST: Local Diagnostics for Posterior Approximations in
Simulation-Based Inference](https://arxiv.org/abs/2306.03580)


Dependencies:
- conda environment: python 3.10
- pip packages: `lampe`, `sbi`, `sbibm`, `seaborn`, `tueplots`, `zuko`

Run `pip install -e .` within the `lc2st` folder. This will automatically install all dependencies.

=================================================================================

Special requirements for the `bernoulli_glm_(raw)` tasks:
- intall the `pypolyagamma` package (or clone it from [this repo](https://github.com/slinderman/pypolyagamma/tree/master) making required changes in the `deps` folder)
- clone and modify the `sbibm` repository: `pip install -e .` in the cloned folder after changing the `task.py` file as follows:
  - add `observation: Optional[torch.Tensor] = None,` as an input variable to the `_sample_reference_posterior` method
  - change `dtype=np.int` to `int` in line 245


## Repoduce Results and Figures

### Experiment 1: Single class evaluation

To generate the Figure 1, run:
```
python figure1_lc2st_2023.py --plot
```

Precomputed are available in `saved_experiments/exp_1`. They were obtained by running the following commands in the terminal:
```
python figure1_lc2st_2023.py --opt_bayes --t_shift
python figure1_lc2st_2023.py --power_shift
```

### Experiment 2: Statistical method comparison on `sbibm` tasks
To generate the figure for each task (i.e. each row in Figures 2 in the main and 5 in Appendix F.1), run:
```
python figure2_lc2st_2023.py --plot --task <task_name>
```
For `task_name = two_moons`, `slcp`, `gaussian_mixture`, `gaussian_linear_uniform`, `bernoulli_glm`, `bernoulli_glm_raw`.

Precomputed are available in `saved_experiments/exp_2`. They were obtained by running the following commands in the terminal:

1. Varying N_train (Columns 1 and 2):
```
python figure2_lc2st_2023.py --task <task_name> --t_res_ntrain --n_train 100 1000 10000 100000
python figure2_lc2st_2023.py --task <task_name> --t_res_ntrain --n_train 100 1000 10000 100000 --power_ntrain
```

2. Varing N_cal (Columns 3 and 4):
```
python figure2_lc2st_2023.py --task <task_name> --power_ncal --n_cal 100 500 1000 2000 5000 10000
```

3. Runtimes (Appendix A.5)
```
python figure2_lc2st_2023.py --task slcp --runtime -nt 0 --n_cal 5000 10000
```

### Experiment 3: Global / Local Coverage Test and graphical diagnostics on the JRNMM
To generate Figure 3, run:
```
python figure3_lc2st_2023.py --plot
```
Add `--lc2st_interpretability` for Figure 4 and Appendix D.

Precomputed are available in `saved_experiments/exp_2`. They were obtained by running the following commands in the terminal:

1. Global Coverage results (left panel of Figure 3):
```
python figure3_lc2st_2023.py --global_ct
```

2. Local Coverage results for varying gain (right panel of Figure 3 and Figure 4 and Appendix D):
```
python figure3_lc2st_2023.py --local_ct_gain
```

### Appendix F.2: Accuracy of L-C2ST w.r.t. the true C2ST
To generate Figure 6 (correlation scatter plots) and results for Table 3 (corresponding p-values), run:
```
python lc2st_stats_scatter_plots_sbibm.py --observations <task/empirical> --method <lc2st/lc2st_nf> --task <task_name>
```
For `task_name = two_moons`, `slcp`, `gaussian_mixture`, `gaussian_linear_uniform`, `bernoulli_glm`, `bernoulli_glm_raw`.

Specifying `empirical` for the `--observations` argument, generates results and plots for `100` different reference observations (instead of just `10`).

Precomputed are available in `saved_experiments/exp_2/<task_name>/scatter_plots`. They were obtained by running the previous command in the terminal.
Add `--sbibm_obs` to generate the new 100 observations via the `sbibm` platform. This can be problematic for some tasks, which is why by default they are simply generated them with the `prior` and `simulator` directly.



