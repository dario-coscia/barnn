import argparse
import os
import shutil

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from barnn.mol.model import RNN
from barnn.mol.module import LanguageModel, BARNNLanguageModel
from barnn.mol.dataset import TDCDataModule
from barnn.mol.callbacks import MoleculeEvaluationCallback

# Main function
def main(args):
 
    # Set the seed
    pl.seed_everything(args.seed, workers=True, verbose=False)

    # Initialize data module
    dataset_folder = None if args.dataset_folder.lower() == 'none' else args.dataset_folder
    dm = TDCDataModule(batch_size=args.batch_size, dataset_folder=dataset_folder)
    dm.initialize()

    # Initialize model and module
    model = RNN(
        vocab_size=len(dm.vocabulary),
        hidden_size=args.hidden_size,
        layer_type=args.solver,
        num_layers=args.num_layers
    )
    if args.run_type == 'train':
        try:
            shutil.rmtree(f'ExperimentMol/{args.solver}/seed_{args.seed}/lightning_logs/')
        except FileNotFoundError:
            pass
        if args.solver == 'barnn':
            module = BARNNLanguageModel(model, dm.vocabulary, dm.tokenizer)
        else:
            module = LanguageModel(model, dm.vocabulary, dm.tokenizer)
    elif args.run_type == 'test':
        ckpt_path = f"ExperimentMol/{args.solver}/seed_{args.seed}/lightning_logs/version_0/checkpoints/checkpoint.ckpt"
        if args.solver == 'barnn':
            module = BARNNLanguageModel.load_from_checkpoint(
                ckpt_path, model=model, vocabulary=dm.vocabulary,
                tokenizer=dm.tokenizer)
        else:
            module = LanguageModel.load_from_checkpoint(
                ckpt_path, model=model, vocabulary=dm.vocabulary,
                tokenizer=dm.tokenizer)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.device,
        devices=args.devices,
        num_nodes=args.num_nodes,
        callbacks=[MoleculeEvaluationCallback(args.solver),
                   ModelCheckpoint(filename='checkpoint')],
        logger=TensorBoardLogger(save_dir=f'ExperimentMol/{args.solver}/seed_{args.seed}'),
        enable_progress_bar=True,
        default_root_dir=f'ExperimentMol/{args.solver}/seed_{args.seed}',
        gradient_clip_val=3,
        gradient_clip_algorithm="value",
    )

    # Train (or finetune) the model
    if args.run_type == 'train':
        trainer.fit(model=module, datamodule=dm)
    elif args.run_type == 'test':
        trainer.predict(model=module, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performing Molecule Experiments')
    parser.add_argument('--hidden_size', default=1024, type=int, help='Hidden size of the RNN')
    parser.add_argument('--num_layers', default=3, type=int, help='Number of layers in the RNN')
    parser.add_argument('--solver', default='barnn', type=str, choices=['lstm', 'dropout_lstm', 'barnn'], help='Type of RNN layer')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
    parser.add_argument('--dataset_folder', default='data/mol/ChemBL', type=str,  help='Path to the dataset folder. Set to "none" for no dataset')
    parser.add_argument('--max_epochs', default=12, type=int,  help='Maximum number of epochs to train')
    parser.add_argument('--seed', default=111, type=int, help='Experiment seed')
    parser.add_argument('--device', default='gpu', type=str, choices=['cpu', 'gpu'], 
                        help='Type of accelerator to use (cpu, gpu, tpu, etc.)')
    parser.add_argument('--devices', default=int(os.environ.get('SLURM_NTASKS_PER_GPU', 1)), type=int, help='Number of devices (e.g., GPUs) to use')
    parser.add_argument('--num_nodes', default=int(os.environ.get('SLURM_NNODES',1)), type=int,  help='Number of nodes for distributed training')
    parser.add_argument('--run_type', type=str, choices=['train', 'test'],
                        help='Run type.', default='train')
    args = parser.parse_args()
    main(args)
