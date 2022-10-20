# Sampling on Directed Graph via Gershgorin Disc Alignment

Codes for "Efficient Sampling on Directed Graphs via Gershgorin Disc Alignment."

## Install
Please ensure `conda` (alternatively `miniconda`) has been installed.

1. Create environment.
```bash
$ conda create -n gda python=3.8
```
2. Install requirements.
```bash
$ conda activate gda
$ cd <path-to-this-repo>/GDADirect
$ pip install -r requirements.txt
```
If the above installtion fails, one should manually install required packages as follows:
```bash
$ pip install intel-numpy==1.21.4 intel-scipy==1.7.3 matplotlib==3.6.1 cvxopt==1.3.0 networkx==2.8.7
```

**Note**: 
- `numpy` and `scipy` in our environment is based on `mkl` by Intel. Those based on `OpenBLAS` or `BLAS` may produce different results in terms of running time.
- Our experiments are conducted on Ubuntu 18.04. Results on other operation systems, e.g., Windows 10, can be different.

## Run experiments

- Reconstruction MSE experiments
```bash
# low pass signal
python recon_mse_exp.py --size=200 --g_type=random --s_type=lowpass --topk_ratio=0.1 --k_start=10 --k_end=60 --k_step=10 --graph_rep=5 --sig_rep=3000 --sigma=0.0 --mu=0.001 --para=20 --epr=0.01

# GMRF signal
python recon_mse_exp.py --size=200 --g_type=random --s_type=normal --omega=0.1 --k_start=10 --k_end=60 --k_step=10 --graph_rep=5 --sig_rep=3000 --sigma=0.0 --mu=0.001 --para=20

# Diffusion signal
python recon_mse_exp.py --size=200 --g_type=random --s_type=diffusion --T=50 --alpha=0.1 --k_start=10 --k_end=60 --k_step=10 --graph_rep=5 --sig_rep=3000 --sigma=0.0 --mu=0.001 --para=20 --epr=0.01
```
Note that some hyper-parameters, e.g., `epr`, should be fine-tuned for the best results.

- Running time experiments
```bash
python runtime_exp.py --n_list 100 200 300 400 500 600 700 800 --k_ratio=0.3 --graph_rep=5 --mu=0.001 --p=0.1
```

