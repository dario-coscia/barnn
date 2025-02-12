import torch
from .layers import (
    ARDVariationalDropoutConv1d, ARDVariationalDropoutConv2d, _BaseBARNN)
from pina.solvers import SupervisedSolver
from diffusers.schedulers import DDPMScheduler
from .utils import positional_embedding


class AutoregressiveSolver(SupervisedSolver):
    """
    A simple autoregressive solver class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_history = self.neural_net.torchmodel.time_history

    def forward(self, x, time):
        """
        Forward pass for the solvers, time is used only in variational
        models for the dropout encoding.
        """
        return self.neural_net.torchmodel(x, time)
    
    def preprocess_input_unroll(self, input):
        return input.as_subclass(torch.Tensor)

    @torch.no_grad()
    def unroll(self, input, unrollings):
        """
        Unrolling the solver from the initial input for unrollings steps,
        keeping trajectory on CPU to save GPU memory.
        """
        # Preprocess the input
        input = self.preprocess_input_unroll(input)
        # Initialize the trajectory with the first input (move it to CPU)
        trajectory = [input.to('cpu').squeeze(-1)]  # Move initial input to CPU [batch, 1, ...]
        # Unroll for the specified number of steps
        for idx in range(unrollings):
            # Time for this step
            time = torch.cat([step + idx * self.time_history + torch.zeros(size=(input.shape[0], 1), device=input.device) for step in range(self.time_history)], dim=1)
            # Perform forward computation on GPU
            du_hat_t = self.forward(trajectory[-1].to(input.device), time)
            # Compute the next state and move it to CPU before appending
            next_state = trajectory[-1] + du_hat_t.to('cpu')
            trajectory.append(next_state)
        # Return the entire trajectory as a tensor on the CPU
        return torch.cat(trajectory, 1).unsqueeze(-1)

    def extract_data(self, input_pts, output_pts):
        input_pts = input_pts.squeeze(-1).as_subclass(torch.Tensor)
        batch_size, seq_len = input_pts.shape[0], input_pts.shape[1]
        random_t = torch.randint(0, seq_len - 2*self.time_history + 1, (batch_size, 1), device=input_pts.device)
        t_start = torch.cat([random_t + idx for idx in range(self.time_history)], dim=-1)
        t_end = t_start + self.time_history
        batch_indeces = torch.arange(batch_size).unsqueeze(-1)
        u_start = input_pts[batch_indeces, t_start, ...]
        u_final = input_pts[batch_indeces, t_end, ...]
        return u_start, u_final, t_start, t_end, u_final - u_start
    
    def loss_data(self, input_pts, output_pts):
        """
        Computing the variational loss for an autoregressive solver. The 
        output have shape [B, Nt, Nx, D], input_pts are not used but kept for
        consistency.
        """
        # Extract a time interval
        u_start, _, t_start, _, du = self.extract_data(input_pts, output_pts)
        # Forward pass
        du_hat = self.forward(u_start, t_start)
        # Compute loss
        mse_loss = (du_hat - du).pow(2).mean()
        self.log("mse_loss", float(mse_loss), prog_bar=True, logger=True)
        return mse_loss

class EnsambleSolver(AutoregressiveSolver):
    """
    An autoregressive solver made for ensemble neural operators.
    """
    def __init__(self, *args, mc_steps=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_steps = mc_steps

    def unroll(self, input, unrollings):
        # in ensemble method we perform multiple times single unrolling
        solutions = torch.stack(
            [super(EnsambleSolver, self).unroll(input, unrollings) for _ in range(self.mc_steps)],
            dim=0
        )
        return solutions.mean(dim=0), solutions.var(dim=0)

######## Derived Classes #########
class InputPerturbationSolver(EnsambleSolver):
    """
    Input Perturbation solver class.
    """
    def __init__(self, *args, perturbation = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturbation = perturbation 

    def preprocess_input_unroll(self, input):
        input = input.as_subclass(torch.Tensor)
        perturbation = (self.perturbation * input.abs().max() * 
                        torch.randn_like(input))
        return input + perturbation
    
    def extract_data(self, input_pts, output_pts):
        u_start, u_final, t_start, t_end, du = super().extract_data(
            input_pts, output_pts)
        perturbation = (self.perturbation * u_start.abs().max() * 
                        torch.randn_like(u_start))
        u_start = u_start + perturbation
        return u_start, u_final, t_start, t_end, du


class BARNNPDESolver(EnsambleSolver):
    def forward(self, x, time):
        alpha = self.neural_net.torchmodel.encoding_alpha(x, time)
        return self.neural_net.torchmodel(x, alpha)
    
    def kl_loss(self, alpha):
        """
        Computing the KL loss.
        """
        kl = []
        idx = 0
        for _, module in self.neural_net.named_modules():
            if isinstance(module, _BaseBARNN):
                kl.append(module.kl(alpha[idx]))
                idx += 1
        return sum(kl)

    def loss_data(self, input_pts, output_pts):
        """
        Computing the variational loss for an autoregressive solver. The 
        output have shape [B, Nt, Nx, D], input_pts are not used but kept for
        consistency.
        """
        # Extract a time interval
        u_start, u_final, t_start, t_end, du = super().extract_data(
            input_pts, output_pts)
        # Sample weights and compute KL
        alpha = self.neural_net.torchmodel.encoding_alpha(u_start, t_start)
        kl_loss = self.kl_loss(alpha).mean()
        # Sample state and compute MSE
        du_hat = self.neural_net.torchmodel(u_start, alpha)
        mse_loss = (du_hat - du).pow(2).mean() 
        # Logging
        self.log("mse_loss", float(mse_loss), prog_bar=True, logger=True)
        self.log("kl_loss", float(kl_loss), prog_bar=True, logger=True)
        # Return loss
        return mse_loss + kl_loss

class ARDVariationalSolver(EnsambleSolver):
    def kl_loss(self):
        """
        Computing the KL loss.
        """
        kl = []
        idx = 0
        for _, module in self.neural_net.named_modules():
            if isinstance(module, (ARDVariationalDropoutConv1d, ARDVariationalDropoutConv2d)):
                kl.append(module.kl())
                idx += 1
        return sum(kl)

    def loss_data(self, input_pts, output_pts):
        """
        Computing the variational loss for an autoregressive solver. The 
        output have shape [B, Nt, Nx, D], input_pts are not used but kept for
        consistency.
        """
        # Extract a time interval
        u_start, _, _, _, du = super().extract_data(
            input_pts, output_pts)
        # Sample weights and compute KL
        kl_loss = self.kl_loss()
        # Sample state and compute MSE
        du_hat = self.neural_net.torchmodel(u_start)
        mse_loss = (du_hat - du).pow(2).mean() 
        # Logging
        self.log("mse_loss", float(mse_loss), prog_bar=True, logger=True)
        # Return loss
        return mse_loss + kl_loss
    
class RefinerSolver(EnsambleSolver):
    """
   Refiner solver class.
    """
    def __init__(self, *args, num_steps = 2, min_noise_std = 2e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps # as in original paper
        self.min_noise_std = min_noise_std
        self.time_multiplier = 1000 / num_steps
        betas = [min_noise_std ** (k / num_steps) for k in reversed(range(num_steps + 1))]
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=num_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )

    def forward(self, x, k):
        k = self.time_multiplier * k
        time_emb = positional_embedding(k, dim=self.neural_net.torchmodel.width)
        return super().forward(x, time_emb).as_subclass(torch.Tensor)
    
    @torch.no_grad()
    def unroll(self, input, unrollings):
        """
        Unrolling the optimizer from the initial input for unrollings steps.
        """
        # Unrolling the trajectories
        sol = []
        for _ in range(self.mc_steps):
            trajectory = [input.as_subclass(torch.Tensor).to('cpu').squeeze(-1)]
            for step in range(unrollings):
                u_next = self.predict_next(trajectory[-1].to(input.device))
                trajectory.append(u_next.to('cpu'))
            sol.append(torch.cat(trajectory, 1))
        sol = torch.stack(sol, dim=0)
        return sol.mean(dim=0).unsqueeze(-1), sol.var(dim=0).unsqueeze(-1)

    def predict_next(self, u_prev):
        du_noised = torch.randn_like(u_prev)
        for step in self.ddpm_scheduler.timesteps:
            k = torch.zeros(size=(u_prev.shape[0],1), dtype=u_prev.dtype, device=u_prev.device) + step
            x_in = torch.cat([u_prev, du_noised], dim=1)
            pred = self.forward(x_in, k)
            du_noised = self.ddpm_scheduler.step(pred, step, du_noised).prev_sample.to(u_prev.device)
        return du_noised + u_prev

    def loss_data(self, input_pts, output_pts):
        """
        Computing the variational loss for an autoregressive solver. The 
        output have shape [B, Nt, Nx, D], input_pts are not used but kept for
        consistency.
        """
        # Extract a time interval
        u_start, _, _, _, du = super().extract_data(
            input_pts, output_pts)
        # Extract refinment steps
        k = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (du.shape[0],1), device=du.device)
        # Add noise
        noise_factor = self.ddpm_scheduler.alphas_cumprod.to(du.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(du.ndim - 1)])
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(du)
        du_noised = self.ddpm_scheduler.add_noise(du, noise, k)
        input_ = torch.cat([u_start, du_noised], dim=1)
        pred = self.forward(input_, k)
        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * du
        # Compute loss
        mse_loss = (pred - target).pow(2).mean()
        self.log("mse_loss", float(mse_loss), prog_bar=True, logger=True)
        return mse_loss