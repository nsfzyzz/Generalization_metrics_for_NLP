# NLP Metrics

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