import os
import lightning.pytorch as pl
import csv
from tdc import Evaluator


class MoleculeEvaluationCallback(pl.callbacks.Callback):
    def __init__(self, model_name, testing_sample_size = 50000,
                 train_sample_size=1000, batch_size=1000):
        super().__init__()
        self.train_sample_size = train_sample_size
        self.testing_sample_size = testing_sample_size
        self.batch_size = batch_size
        self.model_name = model_name
        os.makedirs('ExperimentMol', exist_ok=True)
        self.path = 'ExperimentMol'

        # Generative Model Chemical Ealuation 
        self.single_evaluators = {
            "validity": Evaluator(name='Validity'),
            "diversity": Evaluator(name='Diversity'),
            "uniqueness": Evaluator(name='Uniqueness'),
        }
        # Compute Generative and Dataset Statistics
        self.pair_evaluators = {
            "novelty" : Evaluator(name='Novelty'),
        }

    def sample(self, pl_module, n_samples):
        samples = []
        for _ in range(n_samples // self.batch_size):
            samples+=pl_module.sample(self.batch_size)
        return samples

    def on_train_epoch_end(self, trainer, pl_module):
        # Sample molecules
        smiles = self.sample(pl_module, self.train_sample_size)
        # Evaluate and log metrics
        for name, evaluator in self.single_evaluators.items():
            try:
                score = evaluator(smiles)
            except:
                score = 0
            trainer.logger.log_metrics({name: score}, step=trainer.current_epoch)
            pl_module.log(name, score, prog_bar=True, on_epoch=True)

    def on_predict_start(self, trainer, pl_module):
        """
        Computes test statistics and generates visualizations after model training ends.

        This function performs the following tasks:
        1. Samples a set of generated molecules from the model.
        2. Filters the generated SMILES strings to include only valid ones (expect for Validity, in that case we do not filter).
        3. Computes various evaluation metrics (e.g., validity, diversity, novelty) for the generated molecules.
            Note validity is computed for the raw sampled molecules (before extracting the valid ones).
        4. Saves evaluation results into a CSV file.
        """
        # Sample molecules
        smiles = self.sample(pl_module, self.testing_sample_size)

        # Getting training smiles
        train_smiles_labels = trainer.datamodule.smiles_data["train"]
        train_smiles_tokens = pl_module.vocabulary.labels2tokens(train_smiles_labels)
        train_smiles = [pl_module.tokenizer.untokenize(token) for token in train_smiles_tokens]
        
        # Save statistics in CSV file
        csv_filename = f'{self.path}/{self.model_name}/scores.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['name', 'score'])

            # Single evaluators
            for name, evaluator in self.single_evaluators.items():
                score = evaluator(smiles)
                print(f'{name} : {score}')
                writer.writerow([name, score])

            # Pair evaluators
            for name, evaluator in self.pair_evaluators.items():
                score = evaluator(smiles, train_smiles)
                print(f'{name} : {score}')
                writer.writerow([name, score])
        print(f'Scores saved to {csv_filename}')
