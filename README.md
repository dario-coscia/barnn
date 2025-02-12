<h1 align="center">BARNN: A Bayesian Autoregressive and Recurrent Neural Network - Official Repository </h1> 


<p align="center">
  <a href="https://dario-coscia.github.io/">Dario Coscia</a>
  ,
  <a href="https://staff.fnwi.uva.nl/m.welling/">Max Welling</a>
  ,
  <a href="https://nicolademo.xyz/">Nicola Demo</a>
  ,
  <a href="https://people.sissa.it/~grozza/">Gianluigi Rozza</a>
</p>
<h3 align="center"> <a href="https://www.canva.com/design/DAGdxn0D0CE/_6JRv23f9WTUp2gGCYPsRw/edit?utm_content=DAGdxn0D0CE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton">[Slides]</a> <a href="https://arxiv.org/abs/2501.18665">[Paper]</a>  </h3>

## 📔 Description
BARNN is a cutting-edge technique that enhances autoregressive and recurrent neural networks by incorporating Bayesian inference. This allows for improved uncertainty estimation, scalability, and calibration across a wide range of sequence-based tasks, such as Partial Differential Equations (PDEs) and molecular modeling.

By adopting BARNN, you can:

- Easily convert existing sequence models into Bayesian models.
- Maintain the core architecture with only minimal changes.
- Achieve more reliable uncertainty estimates, making your models more robust and interpretable.

This repository contains the source code to reproduce the experiment for PDEs and molecules, as well as a simple jupyter notebook to start playing around with BARNN.

## 📝 Step-by-Step Guide to build BARNN from scratch
If you're interested in BARNN and want to learn how to implement it from scratch, we’ve created a [Jupyter notebook](./notebooks/example.ipynb) that can be run on Google Colab. In this notebook, we walk through all the essential steps to build BARNN and give you a hands-on introduction to experimenting with it!

## 🛠️ Installation and set up
Clone the git repository, create a virtual conda environment and install the requirements.
```bash
# clone project   
git clone https://github.com/dario-coscia/barnn
cd barnn

# create virtual environment
conda create -n barnn_env python=3.9
conda activate barnn_env

# install project
python -m pip install .   
 ```    

## 💾 Data to reproduce the experiments
Data to reproduce the experiments are available [here](https://huggingface.co/datasets/dario-coscia/BARNN-datasets/tree/main). Download the `data/` folder
and place it in `barnn/.`. 

**Important note:**
The data used in this study were either generated or sourced from previously cited works (see out article), to which full credits are given.

## 💻 Run Experiments
In order to run the experiments we provide two files in the directory `scripts`. The file `scripts/run_pde.py` runs the PDE experiments, while `scripts/run_mol.py` runs the molecules experiments (we suggest using multi-gpu training for this, you can see a SLURM `.sbatch` file inside `shell/`). 

### Reproduce PDE Experiments
**Train the models by**:
```shell
python scripts/run_pde.py --solver={barnn, dropout, refiner, perturb, ard} --pde={Burgers, KS, KdV} --run_type=train
```
Once the model is trained it saves the checkpoints into the directory `ExperimentPDE/{pde}/{solver}/{seed}/lightning_logs/version_0/checkpoints/checkpoint.ckpt`. This checkpoint can be used to load the model to perform UQ analysis. For example, for the BARNN model you can load it by:
```python
path = f'ExperimentPDE/Burgers/barnn/111/lightning_logs/version_0/checkpoints/checkpoint.ckpt'
BARNNPDESolver.load_from_checkpoint(
        checkpoint_path = path,
        problem = problem,
        model = model,
    )
```
**Test the model by**:
```shell
python scripts/run_pde.py --solver={barnn, dropout, refiner, perturb, ard} --pde={Burgers, KS, KdV} --run_type=test
```
The program saves the NLL, ECE, RMSE metrics for each unroll step in `ExperimentPDE/Burgers/{solver}`, and print in std-out the average results on time.

### Reproduce Mol Experiments
**Train the models by**:
```shell
python scripts/run_mol.py --solver={barnn, dropout_lstm, lstm} --run_type=train
```
Once the model is trained it saves the checkpoints into the directory `ExperimentMol/logs_{args.solver}/seed_{args.seed}/lightning_logs/version_0/checkpoints/checkpoint.ckpt`. This checkpoint can be used to load the model to perform drug discovery analysis. For example, for the BARNN model you can load it by:
```python
path = f'ExperimentMol/barnn/111/lightning_logs/version_0/checkpoints/checkpoint.ckpt'
barnn_module = BARNNLanguageModel.load_from_checkpoint(
                    ckpt_path,
                    model=model,
                    vocabulary=dm.vocabulary,
                    tokenizer=dm.tokenizer)
    )
```
You can use the loaded module to sample molecules by `barnn_module.sample(batch_size=30, temperature=1)`, which samples $30$ molecules using temperature=$1$.

**Test the model by**:
```shell
python scripts/run_mol.py --solver={barnn, dropout_lstm, lstm} --run_type=test
```
The program saves the Validity, Novelty, Uniqueness and Diversity metrics in `ExperimentMol/{solver}/`.


### Citation   
```
@article{coscia2025barnn,
  title={{BARNN: A Bayesian Autoregressive and Recurrent Neural Network}},
  author={Coscia, Dario and Welling, Max and Demo, Nicola and Rozza, Gianluigi},
  journal={arXiv preprint arXiv:2501.18665},
  year={2025}
}
```   
