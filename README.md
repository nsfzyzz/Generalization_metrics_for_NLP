# NLP Metrics

For Transformer training, we follow (:link: [Vaswani et al.](https://arxiv.org/abs/1706.03762)). We develop our implementation based on an (:link: [online repository](https://github.com/gordicaleksa/pytorch-original-transformer)).
This code reproduces the results from Vaswani et al. with more easily configurable Transformer architectures.

In addition to the HT-SR theory, we also evaluate generalization metrics from (:link: [Dziugaite et al. 2020.](https://proceedings.neurips.cc/paper/2020/file/86d7c8a08b4aaa1bc7c599473f5dddda-Paper.pdf))


## Setup the environment

### Step 1
Create an environment.
```
conda env create
```
### Step 2
Setup the weightwatcher packages.
```
NLP_metrics/setup_env.sh
```
### Step 3
Copy data.
```
./cp_data.sh
```
### Step 4
Get the new version of weightwatcher from the AWS machine and the file to change the weightwatcher and powerlaw files.

## Generate the experiment commands
```
python generate_batch_scripts/all_experiments.py
```