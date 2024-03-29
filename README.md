# NLP metrics
This repository contains the code to reproduce the results from the paper :link: [Evaluating natural language processing models with generalization metrics that do not need access to any training or testing data.](https://arxiv.org/pdf/2202.02842.pdf) Our main results are that metrics from the :link: [HT-SR theory](https://github.com/CalculatedContent/WeightWatcher) can predict the generalization of NLP models. Also, unlike existing generalization metrics that focus on the "generalization gap", the HT-SR theory can predict the quality of NLP models, e.g., measured by the test-time BLEU scores when the NLP task is neural machine translation.

We mainly study Transformers in this paper. For Transformer training, we follow :link: [Vaswani et al.](https://arxiv.org/abs/1706.03762). We develop our implementation based on an :link: [online repository](https://github.com/gordicaleksa/pytorch-original-transformer). This code reproduces the results from Vaswani et al. with more easily configurable Transformer architectures. In addition to the HT-SR theory, we also evaluate generalization metrics from :link: [Dziugaite et al. 2020.](https://proceedings.neurips.cc/paper/2020/file/86d7c8a08b4aaa1bc7c599473f5dddda-Paper.pdf) and :link: [Jiang et al. 2019.](https://arxiv.org/abs/1912.02178)

## Setup the environment

Step 1. Create a conda environment.
```
conda env create
```
Activate the environment.
```
conda activate NLP_metrics
```

Step 2. Download data and pretrained results.
```
./download_data.sh
```

## Generate the experiment files. Change the checkpoint repository if necessary.
```
python create_experiment.py --CKPT_DIR <your_checkpoint_directory>
```
For example, on my machine, the checkpoint directory is `/data/yyaoqing/Generalization_metrics_for_NLP/checkpoint/`.

## Reproduce the figures shown in paper

### Result 1. Examples of PL fittings.

You can check the examples of PL and E-TPL fittings. Take a look at `visualization/Visualize_example_WW_layers.ipynb`.

<img src="visualization/TPL_vs_PL_mediocre.png" alt="drawing" width="320"/>

### Result 2. Scatter plots.

Then, you can reproduce the scatter plots that compare the generalization metrics with the BLEU scores. Check `visualization/reproduce_scatterplot.ipynb`.

![Block](visualization/Best_ETPL_Lambda.png)

### Result 3. Box plots.

You can also reproduce the box plots that rank the generalization metrics considered in the paper. 

![Block](visualization/Model_quality_vs_generalization_gap.png)

First, use the following commands to generate the time-wise correlations. The argument `--bleu_type` can be used to choose the correlation with the test BLEU scores or the generalization gap.
```
python time_wise_correlation.py --bleu_type test
python time_wise_correlation.py --bleu_type gap
```

Second, Generate the correlation results when a single hyperparameter is varied.
```
python aggregate_hyperparameter_correlation.py
```

Now, you should have all the results. Check `visualization/calculate_rank_correlation_with_colored_groups.ipynb` to see the box plots.

## Reproduce all the training results.

Fully reproducing our results requires :link: [slurm](https://slurm.schedmd.com/) and about 6T storage.

Step 1. Generate slurm configuration files. Check the `scripts/generate_script.ipynb` to generate the training and evaluation slurm configrations.

Step 2. Submit the slurm files. Remember to change the directories in the slurm file and make a slurm log folder.
```
mkdir slurm_logs
```

For training, do the following.
```
sbatch ./scripts/slurm_train_models.sh
```
For evaluation, use the following bash files.
```
sbatch ./scripts/slurm_eval_bleu.sh
sbatch ./scripts/slurm_compute_ww.sh
sbatch ./scripts/slurm_robust_measures.sh
```
Notice that we evaluate PL, E-TPL and EXP fittings. To select the distribution, change L23-33 in the file `slurm_compute_ww.sh`.

Step 3. After generating all the evaluation files, you will get all the json and pickle files similar to the `checkpoint.zip`. Then, you can draw the scatter plots and calculate the rank correlations using the following commands.
```
./scripts/run_plot_scatterplot.sh
./scripts/run_hyperparameter_correlation.sh
```
After that, you will get all the plots and rank correlation results similar to the `plots.zip` and `results.zip`.

## Citation

We appreciate it if you would please cite the following paper if you found the repository useful for your work:

```
@TECHREPORT{yang2022evaluating,
  author =       {Yang, Yaoqing and Theisen, Ryan and Hodgkinson, Liam and Gonzalez, Joseph E and Ramchandran, Kannan and Martin, Charles H and Mahoney, Michael W},
  title =        {Evaluating natural language processing models with generalization metrics that do not need access to any training or testing data},
  number =       {Preprint: arXiv:2202.02842},
  year =         {2022},
}
```

License
----

MIT
