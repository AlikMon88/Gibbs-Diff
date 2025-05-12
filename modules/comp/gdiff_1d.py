import math
import pytorch as torch

class GDiff(nn.Modules):
    def __init__(self):
        pass

    def blind_denoising(self, y, yt,
                            norm_phi_mode='compact',
                            num_chains_per_sample=1,
                            n_it_gibbs=30,
                            n_it_burnin=10,
                            sigma_min=0.04,
                            sigma_max=0.3,
                            return_chains=True):
            '''Gibbs-Diffusion: performs blind denoising with a Gibbs sampler alternating between the diffusion model step returning a sample from p(x|y,phi) and the HMC step that return estimates of parameters from p(phi|x,y).'''

            num_samples = y.shape[0]
            ps_model = iut.ColoredPS(norm_input_phi=norm_phi_mode)
            
            # Prior, likelihood, and posterior functions
            sample_phi_prior = lambda n: iut.sample_prior_phi(n, norm=norm_phi_mode, device=self.device) # Sample uniformly in [-1, 1]
            log_likelihood = lambda phi, x: iut.log_likelihood_eps_phi_sigma(phi[...,:1], phi[...,1:], x, ps_model)
            log_prior = lambda phi: iut.log_prior_phi_sigma(phi[...,:1], phi[...,1], sigma_min, sigma_max, norm=norm_phi_mode)
            log_posterior = lambda phi, x: log_likelihood(phi, x) + log_prior(phi) #  Log posterior (not normalized by the evidence).

            # Bounds and collision management
            phi_min_norm, phi_max_norm = iut.get_phi_bounds(device=self.device) #change to work in [-1,1]
            phi_min_norm, phi_max_norm = iut.normalize_phi(phi_min_norm, mode=norm_phi_mode), iut.normalize_phi(phi_max_norm, mode=norm_phi_mode) #change to work in [-1,1]
            sigma_min_tensor = torch.tensor([sigma_min]).to(self.device)
            sigma_max_tensor = torch.tensor([sigma_max]).to(self.device)
            phi_min_norm = torch.concatenate((phi_min_norm, sigma_min_tensor)) # Add sigma_min to the list of parameter bounds
            phi_max_norm = torch.concatenate((phi_max_norm, sigma_max_tensor)) # Add sigma_max to the list of parameter bounds

            def collision_manager(q, p, p_nxt):
                p_ret = p_nxt
                for i in range(2):
                    crossed_min_boundary = q[..., i] < phi_min_norm[i]
                    crossed_max_boundary = q[..., i] > phi_max_norm[i]
                    # Reflecting boundary conditions
                    p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
                    p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]
                return p_ret

            #print("Normalized prior bounds are:", phi_min_norm, phi_max_norm)

            # Inference on the noise level \sigma and the parameters \varphi of the covariance of the noise

            # Repeat the data for each chain
            y_batch = y.repeat(num_chains_per_sample, 1, 1, 1)
            yt_batch = yt.repeat(num_chains_per_sample, 1, 1, 1)

            # Initalization
            phi_0 = sample_phi_prior(num_samples*num_chains_per_sample)
            sigma_0 = iut.get_noise_level_estimate(y_batch, sigma_min, sigma_max).unsqueeze(-1) # sigma_0 is initalized with a rough estimate of the noise level
            phi_0 = torch.concatenate((phi_0, sigma_0), dim=-1) # Concatenate phi and sigma

            # Gibbs sampling
            phi_all, x_all = [], []
            phi_all.append(phi_0)
            phi_k = phi_0
            step_size, inv_mass_matrix = None, None
            for n in tqdm(range(n_it_gibbs + n_it_burnin)):

                # Diffusion step
                timesteps = self.get_closest_timestep(phi_k[:, 1])
                x_k = self.denoise_samples_batch_time(yt_batch, timesteps, phi_ps=iut.unnormalize_phi(phi_k[:, :1], mode=norm_phi_mode))
                eps_k = (y_batch - x_k)
                
                # HMC step
                log_prob = lambda phi: log_posterior(phi, eps_k)
                def log_prob_grad(phi):
                    """ Compute the log posterior and its gradient."""
                    phib = phi.clone()
                    phib.requires_grad_(True)
                    log_prob_val = log_posterior(phib, eps_k)
                    grad_log_prob = torch.autograd.grad(log_prob_val, phib, grad_outputs=torch.ones_like(log_prob_val))[0]
                    return log_prob_val.detach(), grad_log_prob
                
                if n == 0:
                    hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)
                    hmc.set_collision_fn(collision_manager)

                    phi_k = hmc.sample(phi_k, nsamples=1, burnin=10, step_size=1e-6, nleap=(5, 15), epsadapt=300, verbose=False, ret_side_quantities=False)[:, 0, :].detach()
                    step_size = hmc.step_size
                    inv_mass_matrix = hmc.mass_matrix_inv
                else:
                    hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)
                    hmc.set_collision_fn(collision_manager)
                    hmc.set_inv_mass_matrix(inv_mass_matrix, batch_dim=True)
                    phi_k = hmc.sample(phi_k, nsamples=1, burnin=10, step_size=step_size, nleap=(5, 15), epsadapt=0, verbose=False)[:, 0, :].detach()

                # Save samples
                phi_all.append(phi_k)
                x_all.append(x_k.detach().cpu())

            phi_all = torch.stack(phi_all, dim=1)
            x_all = torch.stack(x_all, dim=1)
            if return_chains:
                return phi_all, x_all
            else:
                x_all
    
    def blind_denoising_pmean(self,y, yt,
                        norm_phi_mode='compact',
                        num_chains_per_sample=5,
                        n_it_gibbs=30,
                        n_it_burnin=10, 
                        avg_pmean=10,
                        return_chains=True):
        '''Performs blind denoising with the posterior mean estimator.'''
        if return_chains:
            phi_all, x_all = self.blind_denoising(y, yt, norm_phi_mode=norm_phi_mode, num_chains_per_sample=num_chains_per_sample, n_it_gibbs=n_it_gibbs, n_it_burnin=n_it_burnin, return_chains=return_chains)
        else:
            x_all = self.blind_denoising(y, yt, norm_phi_mode=norm_phi_mode, num_chains_per_sample=num_chains_per_sample, n_it_gibbs=n_it_gibbs, n_it_burnin=n_it_burnin, return_chains=return_chains)
        x_denoised_pmean = x_all[:, -avg_pmean:].reshape(num_chains_per_sample, -1, 10, self.img_depth, 256, 256).mean(dim=(0, 2))
        if return_chains:
            return phi_all, x_denoised_pmean
        else:
            return x_denoised_pmean


    def get_closest_timestep(self, noise_level, ret_sigma=False):
        """
        Returns the closest timestep to the given noise level. If ret_sigma is True, also returns the noise level corresponding to the closest timestep.
        """
        alpha_bar_t = self.alpha_bar_t.to(noise_level.device)
        all_noise_levels = torch.sqrt((1-alpha_bar_t)/alpha_bar_t).reshape(-1, 1).repeat(1, noise_level.shape[0])
        closest_timestep = torch.argmin(torch.abs(all_noise_levels - noise_level), dim=0)
        if ret_sigma:
            return closest_timestep, all_noise_levels[closest_timestep, 0]
        else:
            return closest_timestep

    def denoise_samples_batch_time(self, noisy_batch, timesteps, batch_origin=None, return_sample=False, phi_ps=None):
        """
        Denoises a batch of images for a given number of timesteps (which can be different across the batch).
        """
        max_timesteps = torch.max(timesteps)
        mask = torch.ones(noisy_batch.shape[0], max_timesteps+1).to(self.device)
        for i in range(noisy_batch.shape[0]):
            mask[i, timesteps[i]+1:] = 0

        for t in range(max_timesteps, 0, -1):
            noisy_batch = self.denoise_1step(noisy_batch, torch.tensor(t).cuda(), phi_ps) * (mask[:, t]).reshape(-1,1,1,1) + noisy_batch * (1 - mask[:, t]).reshape(-1,1,1,1)
        if batch_origin is None:
            return noisy_batch
        else:
            if return_sample:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), noisy_batch
            else:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), None
