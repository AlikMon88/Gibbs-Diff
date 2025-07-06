import numpy as np
import torch
from tqdm import tqdm
import math
from autograd import grad
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# User's existing functions (some might be replaced or become v2)
sigma_eps = 1e-6 # Unused in this snippet, but kept from original
SIGMA_MIN, SIGMA_MAX = 1e-3, 1.0
PHI_MIN, PHI_MAX = -1.0, 1.0

def get_phi_all_bounds(phi_min=PHI_MIN, phi_max=PHI_MAX, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, device = 'cpu'):
    phi_min = torch.tensor([phi_min], device=device)
    phi_max = torch.tensor([phi_max], device=device)
    sigma_min = torch.tensor([sigma_min], device=device)
    sigma_max = torch.tensor([sigma_max], device=device)

    phi_min_all = torch.cat([phi_min, sigma_min])
    phi_max_all = torch.cat([phi_max, sigma_max])
    return phi_min_all, phi_max_all

def get_noise_estimate_2d(y, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    y_std = y.std()
    sigma_est = 1.15 * y_std - 0.17  # heuristic from Imagenet
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est.unsqueeze(0)

def high_pass_filter(y, kernel_size=31):
    # Create a moving average (low-pass) kernel
    kernel = torch.ones(kernel_size) / kernel_size
    kernel = kernel.to(y.device).unsqueeze(0).unsqueeze(0)  # (1, 1, K)
    
    low_pass = F.conv1d(y, kernel, padding=kernel_size // 2)
    high_pass = y - low_pass
    return high_pass.squeeze()

def get_noise_estimate_1d(y, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, kernel_size=31):
    high_freq_part = high_pass_filter(y, kernel_size)
    sigma_est = high_freq_part.std()
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est.unsqueeze(0)

def normalize_phi(phi, phi_min=PHI_MIN, phi_max=PHI_MAX):
    
    """ Normalize phi from its bounded domain to 
    - [0, 1]x[0, 1] for mode=='compact'
    - [-inf, inf]x[-inf, inf] for mode=='inf' """
    
    dphi = phi_max - phi_min
    norm_phi = (phi - phi_min) / dphi
    
    return norm_phi
    
def unnormalize_phi(phi, phi_min=PHI_MIN, phi_max=PHI_MAX):
    
    """ Unnormalize phi according to the prescribed mode."""
    dphi = phi_max - phi_min
    return phi * dphi + phi_min

def sample_phi_prior(n, phi_min=PHI_MIN, phi_max=PHI_MAX, device=None):
    """
    Sample from the prior distribution on phi.
    """
    phi = torch.rand(n).to(device) * (phi_max - phi_min) + phi_min
    return normalize_phi(phi)

def log_prior_phi_sigma(phi_all, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    
    # Unpack parameters
    phi = phi_all[:, 0].unsqueeze(-1) # Shape [B, 1]
    sigma = phi_all[:, 1].unsqueeze(-1) # Shape [B, 1]

    logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float()) #gives either 0 or -inf
    for i in range(1, phi.shape[-1]):
        logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())

    logp += torch.log(torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float()).squeeze(-1)
    return logp


def log_likelihood_eps_phi_sigma(phi_all, eps, ps_model):
    
    # Unpack parameters
    phi = phi_all[:, 0].unsqueeze(-1) # Shape [B, 1]
    sigma = phi_all[:, 1].unsqueeze(-1)         # Shape [B, 1]
    
    ps = ps_model(phi)

    if eps.ndim == 3:  # 1D case
        eps_dim = eps.shape[-1]
        xf = torch.fft.fft(eps)
        sigma = sigma.view(-1, 1) if sigma.ndim == 1 else sigma  # (B, 1)
        scaled_ps = sigma**2 * ps
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=(-1, -2))
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=(-1, -2)) / eps_dim

    elif eps.ndim == 4:  # 2D image case
        H, W = eps.shape[-2], eps.shape[-1]
        eps_dim = H * W
        xf = torch.fft.fft2(eps)
        sigma = sigma.view(-1, 1, 1, 1) if sigma.ndim == 1 else sigma
        scaled_ps = sigma**2 * ps
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=(-1, -2, -3))
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=(-1, -2, -3)) / eps_dim

    else:
        raise ValueError("eps must be 2D (1D case) or 4D (image case)")
    
    log_likelihood = term_pi + term_logdet + term_x  # (b, dim)
    return log_likelihood



class ColoredPowerSpectrum2D(nn.Module):
    def __init__(self, shape = (3, 256, 256), device='cpu', sigma_eps=1e-6):
        super().__init__()
        shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape, )
        ndim = len(shape) - 1

        N = shape[-1]

        assert len(shape) == 3 # (C, H, W)
        assert shape[1] == shape[2] # (C, H, W)

        # Build an array of isotropic wavenumbers
        wn = torch.fft.fftfreq(N).reshape((N, 1)).to(device)

        S = torch.zeros((1,)+shape).to(device)
        for i in range(ndim):
            S += torch.moveaxis(wn, 0, i).pow(2)
        self.S = torch.sqrt(S)

        self.S[:,:, 0, 0] = 1
        
    def forward(self, phi):
        '''Generates a power spectrum S(k) ~ k^alpha
        alpha: tensor of alpha of size (batch_size, phi_dim)
        '''
        phi = unnormalize_phi(phi)
        
        S = self.S.repeat(phi.shape[0], 1, 1, 1)
        S = torch.pow(S, phi.reshape(-1, 1, 1, 1))
        S = S/torch.mean(S, dim=(2, 3), keepdim=True)
        return S

class ColoredPowerSpectrum1D(nn.Module):
    def __init__(self, shape=(1, 100), device='cpu'):
        super().__init__()
        shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        assert len(shape) == 2  # (dim, seq_len)
        
        dim, N = shape
        self.device = device # Added device to constructor

        # Create isotropic wavenumber vector for 1D
        wn = torch.fft.fftfreq(N).to(torch.float32)  # shape: (N,)
        wn = wn.to(device)

        # Compute wavenumber magnitude (squared)
        S = wn.pow(2).reshape(1, 1, N)  # (1, 1, N)
        S = torch.sqrt(S)
        S[:, :, 0] = sigma_eps  # avoid division by zero for k=0

        self.S = S  # shape: (1, 1, N)

    def forward(self, phi):
        '''
        phi: tensor of shape (batch_size, dim) or (batch_size, 1)
             controls the spectral slope alpha
        '''
        phi = unnormalize_phi(phi)

        batch_size, phi_dim = phi.shape
        # Assuming single phi controls all dims
        S = self.S ** phi.view(batch_size, 1, 1) # (B, 1, N)
        S = S / S.mean(dim=-1, keepdim=True)    # Normalize spectrum
        return S

class DualAveragingStepSize():
    """ Dual averaging step size adaptation (Nesterov 2009). """

    def __init__(self, initial_step_size,
                 target_accept=0.65,
                 gamma=0.05,
                 t0=10.0,
                 kappa=0.75,
                 nadapt=0):
        """Constructor

        Args:
            initial_step_size (torch.Tensor): Initial step size.
            target_accept (float, optional): Target Metropolis acceptance rate. Must be between 0 and 1. Defaults to 0.65.
            gamma (float, optional): Adaptation regularization scale. Defaults to 0.05.
            t0 (float, optional): Adaptation iteration offset. Defaults to 10.0.
            kappa (float, optional): Adaptation relaxation exponent. Defaults to 0.75.
            nadapt (int, optional): _description_. Defaults to 0.
        """
        self.initial_step_size = initial_step_size 
        self.mu = torch.log(initial_step_size) # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma * 2 #parameter to tune
        self.t = t0
        self.kappa = kappa
        self.error_sum = torch.zeros_like(self.initial_step_size).to(initial_step_size.device)
        self.log_averaged_step = torch.zeros_like(self.initial_step_size).to(initial_step_size.device)
        self.nadapt = nadapt
        
    def update(self, p_accept):
        p_accept[p_accept > 1] = 1.
        p_accept[torch.isnan(p_accept)] = 0.
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return torch.exp(log_step), torch.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n" % step_size)
        else:
            step_size = torch.exp(self.log_averaged_step)
        return step_size
    


class HMC():
    """ Hamiltonian Monte Carlo Sampler"""

    def __init__(self, log_prob,
                 grad_log_prob=None,
                 log_prob_and_grad=None,
                 inv_mass_matrix=None,
                 precision=torch.float32):
        """Constructor

        Args:
            log_prob (_type_): Function that returns the log probability of the target distribution.
            grad_log_prob (_type_, optional): Function that returns the gradient of the log probability of the target distribution. Defaults to None.
            log_prob_and_grad (_type_, optional): Function that returns both the log probability of the target distribution and its gradient. Defaults to None.
            inv_mass_matrix (_type_, optional): Inverse mass matrix. Defaults to None (i.e. identity).
            precision (_type_, optional): Float precision. Defaults to torch.float32.
        """

        self.precision = precision
        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad
        
        assert not((self.grad_log_prob is None) and (self.log_prob_and_grad is None))

        # Convert to prescribed precision
        if self.log_prob is not None:
            self.log_prob = lambda x: log_prob(x).to(self.precision)
        if self.grad_log_prob is not None:
            self.grad_log_prob = lambda x: grad_log_prob(x).to(self.precision)
        if self.log_prob_and_grad is not None:
            self.log_prob_and_grad = lambda x: tuple([y.to(self.precision) for y in log_prob_and_grad(x)])

        # Set inverse mass matrix and define corresponding kinetic energy function and its gradient
        self.set_inv_mass_matrix(inv_mass_matrix)

        # Define the potential energy
        self.V = lambda x: -self.log_prob(x)

        # Collision function
        self.collision_fn = None

        # Counters
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0
    
    def reset_counters(self):
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0
    
    def set_inv_mass_matrix(self, mat, batch_dim=False):
        """ Sets the inverse mass matrix and defines corresponding kinetic energy function and its gradient.

        Args:
            mat (torch.Tensor or None): Diagonal elements or full matrix of the inverse mass matrix.
            batch_dim (bool, optional): Is there a batch dimension? Defaults to False.
        """
        if mat is None:
            self.mass_matrix_inv = None
            self.mass_matrix_sqrt = None

            # Define kinetic energy and its gradient
            self.KE = lambda p: 0.5*(p**2).sum(-1) # Sum across parameter dimension
            self.KE_g = lambda p: p
        else:
            assert mat.ndim == 1 + int(batch_dim) or mat.ndim == 2 + int(batch_dim)
            if batch_dim:
                self.mass_matrix_batch_dim = True
            if mat.ndim == 1 + int(batch_dim):
                self.mass_matrix_inv = mat
                self.mass_matrix_sqrt = torch.stack([torch.diag(m ** -0.5) for m in mat], dim=0)

                # Define kinetic energy and its gradient
                self.KE = lambda p: 0.5*(p**2 * self.mass_matrix_inv).sum(-1) # Sum across parameter dimension
                self.KE_g = lambda p: p * self.mass_matrix_inv
            else:
                L_list, Q_list = [], []
                mass_matrix_sqrt_list = []

                for i,m in enumerate(mat):
                    assert (m == m.T).all() # Check if symmetric
                    L, Q = torch.linalg.eigh(m)
                    if (L > 0).all(): # Check if positive definite
                        L_list.append(L)
                        Q_list.append(Q)
                        mass_matrix_sqrt_list.append(Q @ torch.diag(L**-0.5) @ Q.T)
                    else:
                        m = mat.mean(dim=0)
                        L, Q = torch.linalg.eigh(m)
                        L_list.append(L)
                        Q_list.append(Q)
                        mass_matrix_sqrt_list.append(Q @ torch.diag(L**-0.5) @ Q.T)
                        mat[i] = m
                
                self.mass_matrix_inv = mat
                self.mass_matrix_sqrt = torch.stack(mass_matrix_sqrt_list, dim=0)

                # Define kinetic energy and its gradient
                self.KE = lambda p: 0.5*(p * (self.mass_matrix_inv @ p.unsqueeze(-1)).squeeze(-1)).sum(-1) # Sum across parameter dimension
                self.KE_g = lambda p: (self.mass_matrix_inv @ p.unsqueeze(-1)).squeeze(-1)

    def V_g(self, q):
        """Returns the gradient of the potential energy.

        Args:
            q (torch.Tensor): Position vector.

        Returns:
            torch.Tensor: Gradient of the potential energy at q.
        """
        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(q)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(q)
        return -v_g.detach()

    def H(self, q, p, Vq=None):
        """Returns the Hamiltonian.

        Args:
            q (torch.Tensor): Position vector.
            p (torch.Tensor): Momentum vector.
            Vq (torch.Tensor, optional): Potential energy at q. Defaults to None.

        Returns:
            torch.Tensor: Hamiltonian at (q, p).
        """
        if Vq is None:
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)

    def set_collision_fn(self, collision_fn):
        """Sets custom collision management function for leapfrog updates.

        A collision management function takes as input the position and momentum vectors and returns
        the updated position and momentum vectors.

        Args:
            collision_fn (_type_): Collision function.
        """
        self.collision_fn = collision_fn

    def leapfrog(self, q, p, nleap, step_size):
        """Leapfrog integrator.

        Args:
            q (torch.Tensor): Position vector.
            p (torch.Tensor): Momentum vector.
            nleap (int or (int, int)): Number of leapfrog steps (int), or range for the random draw of the number of leapfrog steps (tuple of (int, int)).
            step_size (torch.Tensor): Step sizes per chain.

        Returns:
            (torch.Tensor, torch.Tensor): Updated (q, p).
        """
        self.leapcount += 1
        s = step_size.unsqueeze(-1)

        assert isinstance(nleap, int) or isinstance(nleap, tuple)
        if isinstance(nleap, tuple):
            N = np.random.randint(nleap[0], nleap[1]+1)
        else:
            N = nleap
 
        p = p - 0.5 * s * self.V_g(q)
        for i in range(N - 1):
            q = q + s * self.KE_g(p)
            if self.collision_fn is not None:
                p = self.collision_fn(q, p, p - s * self.V_g(q))
            else:
                p = p - s * self.V_g(q)
        q = q + s * self.KE_g(p)
        p = p - 0.5 * s * self.V_g(q)
        return q, p
        
    def metropolis(self, q0, p0, q1, p1, V_q0=None, V_q1=None):
        """Metropolis-Hastings acceptance step.

        Args:
            q0 (torch.Tensor): Position vector at step n.
            p0 (torch.Tensor): Momentum vector at step n.
            q1 (torch.Tensor): Position vector at step n+1.
            p1 (torch.Tensor): Momentum vector at step n+1.
            V_q0 (torch.Tensor, optional): Potential energy at q0. Defaults to None.
            V_q1 (torch.Tensor, optional): Potential energy at q1. Defaults to None.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): Position, momentum, acceptance rate,
            Hamiltonians at (q0, p0) and (q1, p1) after the acceptance step.
        """
        
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = torch.exp(H0 - H1)

        u = torch.rand(prob.shape[0], device=prob.device)

        qq = q1.clone()
        pp = p1.clone()
        acc = torch.ones_like(prob)

        cond1 = torch.logical_or(torch.isnan(prob), torch.isinf(prob))
        cond1 = torch.logical_or(cond1, torch.sum(q0 - q1, dim=-1) == 0)

        qq[cond1] = q0[cond1]
        pp[cond1] = p0[cond1]
        acc[cond1] = -1.0

        cond2 = torch.logical_and(u > torch.min(torch.ones_like(u), prob), ~cond1)
        qq[cond2] = q0[cond2]
        pp[cond2] = p0[cond2]
        acc[cond2] = 0.0
        
        return qq, pp, acc, torch.stack([H0, H1], dim=-1)

    def step(self, q, nleap, step_size):
        """Performs a single HMC step.

        Args:
            q (torch.Tensor): Position vector.
            nleap (int or (int, int)): Number of leapfrog steps (int), or range for the random draw of the number of leapfrog steps (tuple of (int, int)).
            step_size (torch.Tensor): Step sizes per chain.

        Returns:
            (torch.Tensor,)*5: Position, momentum, acceptance rate,
            Hamiltonians at (q0, p0) and (q1, p1), misc counts after HMC step.
        """
        p = torch.randn(q.shape, device=q.device, dtype=self.precision)
        if self.mass_matrix_sqrt is not None:
            p = (self.mass_matrix_sqrt @ p.unsqueeze(-1)).squeeze(-1)
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis(q, p, q1, p1)
        return q, p, accepted, Hs, torch.tensor([self.Hcount, self.Vgcount, self.leapcount])

    def adapt_stepsize(self, q, step_size, epsadapt, nleap, verbose=False):
        """ Dual averaging step size adaptation.

        Args:
            q (torch.Tensor): Position vector
            step_size (torch.Tensor): Intial step size.
            epsadapt (int): Number of iterations for step size adaptation.
            nleap (int or (int, int)): Number of leapfrog steps (int), or range for the random draw of the number of leapfrog steps (tuple of (int, int)).

        Returns:
            (torch.Tensor, torch.Tensor): Updated position vector, step size.
        """
        print("Adapting step size using %d iterations" % epsadapt)
        epsadapt_kernel = DualAveragingStepSize(step_size)

        q_list = [] # We save the positions to estimate the inverse mass matrix at half the iterations
        
        for i in tqdm(range((epsadapt)), disable=not verbose):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            q = q.detach()
            Hs = Hs.detach()
            
            q_list.append(q)

            prob = torch.exp(Hs[...,0] - Hs[...,1])

            if i < epsadapt - 1:
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt - 1:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)

            # Estimate the inverse mass matrix
            if i == 3*epsadapt//4:
                q_list_tensor = torch.stack(q_list, dim=-2)[:, epsadapt//4:, :]
                assert q_list_tensor.ndim == 3

                # Batched sample covariance
                B, N, D = q_list_tensor.size()
                mean = q_list_tensor.mean(dim=-2, keepdim=True)
                diffs = (q_list_tensor - mean).reshape(B * N, D)
                prods = torch.bmm(diffs.unsqueeze(-1), diffs.unsqueeze(-2)).reshape(B, N, D, D)
                bcov = prods.sum(dim=-3) / (N - 1)  # Unbiased estimate
                self.set_inv_mass_matrix(bcov.detach(), batch_dim=True)
            step_size.detach()
        return q, step_size
    
    def sample(self, q,
               step_size=0.01,
               nsamples=20,
               burnin=10,
               nleap=30,
               skipburn=True,
               epsadapt=0,
               verbose=False,
               ret_side_quantities=False):
        """Performs HMC sampling.

        Args:
            q (torch.Tensor): Position vector. Possible shapes are (nchains, ndim) or (ndim).
            step_size (float, optional): Step size. If no adaptive step size Defaults to 0.01.
            nsamples (int, optional): Number of samples. Defaults to 20.
            burnin (int, optional): Number of burn-in steps. Defaults to 10.
            nleap (int or (int, int)): Number of leapfrog steps per HMC step (int), or range for the random draw of the number of leapfrog steps (tuple of (int, int)).
            skipburn (bool, optional): Should we save burning samples? Defaults to True.
            epsadapt (int, optional): Epsilon adapt parameter. Defaults to 0.
            verbose (bool, optional): Verbose mode. Defaults to False.
            ret_side_quantities (bool, optional): Should we return side quantities? Defaults to False.

        Returns:
            Sampler: Sampler object.
        """
        if q.ndim == 1: q = q.unsqueeze(0) # We add a chain dimension if there is none.
        assert q.ndim == 2, "q must be 2D" # Shape of q is (nchains, ndim)

        self.reset_counters()

        q = q.to(self.precision)
        
        step_size = step_size * torch.ones((q.shape[0]), device=q.device, dtype=self.precision)

        # We store the samples, acceptance rates, Hamiltonian values, and misc counts
        samples_list = []
        accepts_list = []
        Hs_list = []
        counts_list = []

        if epsadapt > 0:
            q, step_size = self.adapt_stepsize(q, step_size, epsadapt, nleap, verbose=verbose)
            self.step_size = step_size
        for i in tqdm(range(nsamples + burnin), disable=not verbose):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            q.detach()
            Hs.detach()
            accepts_list.append(acc)
            if (skipburn and (i >= burnin)) or not skipburn:
                samples_list.append(q)
                Hs_list.append(Hs)
                counts_list.append(count)
            

        # To torch tensors
        samples_list = torch.stack(samples_list, dim=-2)
        accepts_list = torch.stack(accepts_list, dim=-1)
        Hs_list = torch.stack(Hs_list, dim=-2)
        counts_list = torch.stack(counts_list, dim=-2)
        
        if ret_side_quantities:
            return samples_list, accepts_list, Hs_list, counts_list
        else:
            return samples_list