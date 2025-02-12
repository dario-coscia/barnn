import argparse
import torch
import shutil

from pytorch_lightning.callbacks import ModelCheckpoint

from pina import Condition, Trainer
from pina.problem import AbstractProblem

from barnn.pde.utils import create_directories, read_results
from barnn.pde.stats import Statistics
from barnn.pde.dataset import Dataset
from barnn.pde.modules import (
    EnsambleSolver, ARDVariationalSolver,
    BARNNPDESolver, RefinerSolver, InputPerturbationSolver
    )
from barnn.pde.models import FNO1D

_Models = {
    'fno1d' : FNO1D,
    }

_Solvers = {
    'barnn' : BARNNPDESolver,
    'dropout' : EnsambleSolver,
    'ard' : ARDVariationalSolver,
    'refiner' : RefinerSolver,
    'perturb' : InputPerturbationSolver,
    }

def main(args):
    # seed
    torch.manual_seed(args.seed)
    # start experiments
    train_directory = f'data/pde/{args.pde}_train.h5'
    test_directory = f'data/pde/{args.pde}_test.h5'
    path = create_directories('ExperimentPDE', args.seed, args.pde, args.solver)
    # get the data
    data_train = Dataset(train_directory)
    data_test = Dataset(test_directory)
    # define the problem
    class Problem(AbstractProblem):
        input_variables = ['u']
        output_variables = ['u']
        conditions = {'data' : Condition(input_points=data_train.pde,
                                        output_points=data_train.pde)}
    problem = Problem()
    # define model and solver classes
    model = _Models[args.model](
        num_layers=int(args.num_layers),
        width=int(args.hidden_size),
        modality=args.solver,
        time_history=args.time_history,
        )
    
    if args.run_type == 'train':
        try:
            shutil.rmtree(path + 'lightning_logs/')
        except FileNotFoundError:
            pass
        solver = _Solvers[args.solver](
            problem = problem,
            model=model,
            optimizer = torch.optim.Adam,
            optimizer_kwargs = {'weight_decay' : args.weight_decay, 'lr' : args.lr},
            )
        # define trainer and train
        trainer = Trainer(
            solver=solver,
            batch_size=args.batch_size,
            accelerator=args.device,
            max_epochs=args.max_epochs,
            default_root_dir=path,
            enable_progress_bar=True,
            callbacks=[ModelCheckpoint(filename='checkpoint')]
            )
        trainer.train()

    elif args.run_type == 'test':
        solver = _Solvers[args.solver].load_from_checkpoint(
                checkpoint_path = path + 'lightning_logs/version_0/checkpoints/checkpoint.ckpt',
                problem = problem,
                model = model,
                mc_steps=args.mc_steps,
            )
        # compute statistics and plots
        stats = Statistics(time_history=args.time_history)
        solver = solver.to(args.device)
        data_test.pde = data_test.pde.to(args.device)
        # unroll predictions
        mean, var = solver.unroll(input=data_test.pde[:, 0:args.time_history, ...],
                            unrollings=int(data_test.pde.shape[1]-1) // args.time_history)

        # compute stats
        stats.compute_statistics(ground_truth=data_test.pde,
                                mean=mean,
                                var=var,
                                directory=path + f'scores/')
        read_results(path + f'scores/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performing PDE Experiments')
    parser.add_argument('--seed', type=int, default=111,
                        help='Seed to use')
    parser.add_argument('--pde', type=str, default='Burgers',
                        help='Where to retrieve train data')
    parser.add_argument('--model', type=str, default='fno1d',
                        help=f'Model used as PDE solver: {list(_Models.keys())}')
    parser.add_argument('--solver', type=str, default='barnn',
                        help=f'Model used as PDE solver: {list(_Solvers.keys())}')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of samples in each minibatch')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Used device')
    parser.add_argument('--max_epochs', type=int, default=7000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-8, help='Weight decay.')
    parser.add_argument('--mc_steps', type=int, default=100,
                        help='Number of monte carlo step for computing statistics on models.')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of layers of the model.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='hidden_size of the model.')
    parser.add_argument('--time_history', type=int, default=1,
                        help='Time history for temporal bundling.')
    parser.add_argument('--run_type', type=str, choices=['train', 'test'],
                        help='Run type.', default='train')
    args = parser.parse_args()
    
    # run
    main(args)
