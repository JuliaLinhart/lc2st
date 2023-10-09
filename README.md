# lc2st: local Classifier Two-Sample Tests
Official code for [L-C2ST: Local Diagnostics for Posterior Approximations in
Simulation-Based Inference](https://arxiv.org/abs/2306.03580)

Dependencies:
- conda environment: python 3.10
- pip packages: `lampe`, `sbi`, `sbibm`, `seaborn`, `tueplots`, `zuko`

Run `pip install -e .` within the `lc2st` folder. This will automatically install all dependencies.

=================================================================================

Special requirements for the `bernoulli_glm_(raw)` tasks:
- intall the `pypolyagamma` package
- clone and modify the `sbibm` repository: `pip install -e .` in the cloned folder after changing the `task.py` file as follows:
  - add `observation: Optional[torch.Tensor] = None,` as an input variable to the `_sample_reference_posterior` method
  - change `dtype=np.int` to `int` in line 245

## 1. Generate Figures from paper

### Figure 1: Single-class evaluation
```
python figure1_lc2st_2023.py --plot
```

### Figures 2 (main) and 5 (Appendix F.1): Method comparison on SBIBM tasks
```
python figure2_lc2st_2023.py --task <task_name> --plot
```
For `task_name = two_moons`, `slcp`, `gaussian_mixture`, `gaussian_linear_uniform`, `bernoulli_glm`, `bernoulli_glm_raw`.

### Figure 3: Global vs. Local Coverage Test (JRNMM)
```
python figure3_lc2st_2023.py --plot
```

### Figure 4 (and Appendix D): Interpretability of L-C2ST (graphical diagnostics for JRNMM)
```
python figure3_lc2st_2023.py --plot --lc2st_interpretability
```

### Figure 6 and Table 3 (Appendix F.2): Accuracy of L-C2ST w.r.t. the true C2ST (correlation scatter plots)
```
python lc2st_stats_scatter_plots_sbibm.py --task <task_name> --observations <task/empirical> --method <lc2st/lc2st_nf>
```
For `task_name = two_moons`, `slcp`, `gaussian_mixture`, `gaussian_linear_uniform`, `bernoulli_glm`, `bernoulli_glm_raw`.

Specifying `empirical` for the `--observations` argument, generates results and plots for `100` different reference observations (instead of just `10`).

## 2. Reproduce experiment results from paper

### Results for Figure 1
```
python figure1_lc2st_2023.py --opt_bayes --t_shift
```
```
python figure1_lc2st_2023.py --power_shift
```
### Results for Figures 2 (main) and 5 (Appendix F.1)
1. Varying N_train (Columns 1 and 2):
```
python figure2_lc2st_2023.py --task <task_name> --t_res_ntrain --n_train 100 1000 10000 100000
```
```
python figure2_lc2st_2023.py --task <task_name> --t_res_ntrain --n_train 100 1000 10000 100000 --power_ntrain
```
2. Varing N_cal (Columns 3 and 4):
```
python figure2_lc2st_2023.py --task <task_name> --power_ncal --n_cal 100 500 1000 2000 5000 10000
```
For `task_name = two_moons`, `slcp`, `gaussian_mixture`, `gaussian_linear_uniform`, `bernoulli_glm`, `bernoulli_glm_raw`.

### Results of Appendix A.5 (Runtimes)
```
python figure2_lc2st_2023.py --task slcp --runtime -nt 0 --n_cal 5000 10000
```

### Results for Figures 3 and 4 (and Appendix D):
1. Global Coverage results (left panel of Figure 3):
```
python figure3_lc2st_2023.py --global_ct
```
2. Local Coverage results for varying gain (right panel of Figure 3 and Figure 4 and Appendix D):
```
python figure3_lc2st_2023.py --local_ct_gain
```

### Results for Figure 6 and Table 3 (Appendix F.2):
1. 10 reference observations
```
python lc2st_stats_scatter_plots_sbibm.py --task <task_name> --observations task --method <lc2st/lc2st_nf>
```

2. 100 reference observations
```
python lc2st_stats_scatter_plots_sbibm.py --task <task_name> --observations empirical --method <lc2st/lc2st_nf>
```
For `task_name = two_moons`, `slcp`, `gaussian_mixture`, `gaussian_linear_uniform`, `bernoulli_glm`, `bernoulli_glm_raw`.

You can add `--sbibm_obs` to generate new observations via the `sbibm` platform. This can be problematic for some tasks, which is why by default they are simply generated them with the `prior` and `simulator` directly.





