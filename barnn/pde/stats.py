import torch
from pina import LabelTensor
from .utils import check_directory


def icdf(mean, var, p):
    return mean + torch.erfinv(2 * p - 1) * torch.sqrt(2 * var)

class Statistics:
    AVAILABLE_STATISTICS = [
        'ece_statistics',
        'nll_statistics',
        'rmse_statistics',
    ]
    def __init__(self, time_history=1, types='all', per_time=True):
        assert isinstance(per_time, bool)
        self.per_time = per_time
        # get the type of statistics
        if types == 'all':
            self.types = Statistics.AVAILABLE_STATISTICS
        else:
            assert all(t in Statistics.AVAILABLE_STATISTICS for t in types), \
                "types must be in Statistics.AVAILABLE_STATISTICS"
            self.types = types
        self.time_history = time_history

    def compute_statistics(self, ground_truth, mean, var, directory):
        # check directory
        check_directory(directory)
        # get data
        assert isinstance(ground_truth, (torch.Tensor, LabelTensor))
        assert isinstance(mean, (torch.Tensor, LabelTensor))
        assert isinstance(var, (torch.Tensor, LabelTensor))
        assert ground_truth.shape == mean.shape, f'{ground_truth.shape=}, while {mean.shape=}'
        assert ground_truth.shape == var.shape, f'{ground_truth.shape=}, while {var.shape=}'
        # time 0 is the right time
        self.ground_truth = ground_truth.as_subclass(torch.Tensor)[:, self.time_history:].cpu()
        self.mean = mean.as_subclass(torch.Tensor)[:, self.time_history:].cpu()
        self.var = var.as_subclass(torch.Tensor)[:, self.time_history:].cpu()
        # compute
        for type in self.types:
            func = getattr(self, type)
            stat = func()
            torch.save(stat, f'{directory}/{type}.pt')

    def ece_statistics(self, mc_vals=100):
        # [B, Nt, Nx, F]
        probs = torch.rand(mc_vals)
        p_obs = torch.stack(
            [ (
                self.ground_truth <= icdf(self.mean, self.var, p)
               ).float().cpu() for p in probs
            ], dim=0).mean(dim=(1,2,3,4))
        return (probs - p_obs).abs().mean()

    def nll_statistics(self):
        # compute nll statistics: mean over batch, spatial dimension, field values
        nll = 0.5*(torch.log(2. * torch.pi * self.var) + ((self.ground_truth - self.mean).pow(2) / self.var)).mean(dim=(0, 2, 3))
        # mean over time if needed
        nll = nll if self.per_time else nll.mean(dim=-1)
        return nll

    def rmse_statistics(self):
        # compute q statistics: mean over batch, spatial dimension, field values
        rmse = (self.ground_truth - self.mean).pow(2).mean(dim=(0, 2, 3)).sqrt()
        # mean over time if needed
        rmse = rmse if self.per_time else rmse.mean(dim=-1)
        return rmse