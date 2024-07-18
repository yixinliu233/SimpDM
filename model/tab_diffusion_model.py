"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
"""
import torch.nn.functional as F
import torch
import math
import numpy as np
from .utils import *
from torch.nn.functional import mse_loss

eps = 1e-8

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class SimpDM(torch.nn.Module):
    def __init__(
            self,
            num_numerical_features: int,
            denoise_fn,
            num_timesteps=1000,
            gammas=[1,0.8,0.001],
            gaussian_loss_type='mse',
            gaussian_parametrization='x0',
            scheduler='cosine',
            train_mask_rate = 0.3,
            device=torch.device('cpu'),
            ssl_loss_weight=1.0
        ):

        super(SimpDM, self).__init__()

        self.num_numerical_features = num_numerical_features

        self.ssl_loss_weight = ssl_loss_weight

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.train_mask_rate = train_mask_rate

        self.gamma_ms = gammas[0]
        self.gamma_pm = gammas[1]
        self.gamma_gt = gammas[2]
        self.ssl_loss = mse_loss

        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float().to(device)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))

    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        '''
        x_t = \alpha_t ** 0.5 * x_0 + (1 - \alpha_t) ** 0.5 * eps
        '''
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def sample_mask(self, x, miss_mask_data=None):
        rate_of_one = 1 - self.train_mask_rate
        rand_for_mask = torch.rand_like(x)
        if miss_mask_data is not None:
            rand_for_mask += miss_mask_data
        miss_mask_syn = (rand_for_mask > rate_of_one).float()

        if miss_mask_data is not None:
            loss_mask = miss_mask_syn - miss_mask_data
        else:
            loss_mask = miss_mask_syn
        return miss_mask_syn, loss_mask

    def gaussian_q_sample_with_mask(self, x_start, t, masks, noise=None):

        assert noise.shape == x_start.shape
        assert masks['miss_mask_syn_num'].shape == x_start.shape

        x_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
              extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        if self.gamma_ms or self.gamma_pm or self.gamma_gt:
            noise_gamma = torch.randn_like(x_start)

        if self.gamma_ms > 0:
            x_t += self.gamma_ms * noise_gamma * masks['miss_mask_data_num']
        if self.gamma_pm > 0:
            x_t += self.gamma_pm * noise_gamma * masks['loss_mask_num']

        x_t_masked = x_t * masks['miss_mask_syn_num'] + x_start * (1 - masks['miss_mask_syn_num'])

        if self.gamma_gt > 0:
            x_t_masked +=  self.gamma_gt * noise_gamma * (1 - masks['miss_mask_syn_num'])

        return x_t_masked

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]], dim=0)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, mask=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        if self.gaussian_loss_type == 'mse':
            if self.gaussian_parametrization == 'x0':
                loss = (x_start - model_out) ** 2

            elif self.gaussian_parametrization == 'eps':
                loss = (noise - model_out) ** 2
            else:
                raise NotImplementedError

            if mask != None:
                loss = loss * mask
                loss = loss.sum() / mask.sum()
            else:
                loss = mean_flat(loss)

        elif self.gaussian_loss_type == 'kl':
            loss = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]

        return loss
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(self, model_out, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def get_model_input(self, x_num, t=None, pt=None, masks={}):
        data = {}

        if t == None:
            b = x_num.shape[0]
            device = x_num.device
            t, pt = self.sample_time(b, device, 'uniform')

        data['t'] = t
        data['pt'] = pt
        data['mask_num'] = masks['miss_mask_syn_num']

        noise = torch.randn_like(x_num)
        noise = noise * data['mask_num']
        x_num_t = self.gaussian_q_sample_with_mask(x_num, t, masks=masks, noise=noise)
        data['noise'] = noise
        data['x_num_t'] = x_num_t

        return data

    def train_iter(self, x, miss_mask_data):

        x_num = x
        miss_mask_data_num = miss_mask_data

        masks = {}
        if x_num.shape[1] > 0:
            miss_mask_syn_num, loss_mask_num = self.sample_mask(x_num, miss_mask_data_num)
            masks['miss_mask_syn_num'] = miss_mask_syn_num
            masks['loss_mask_num'] = loss_mask_num
            masks['miss_mask_data_num'] = miss_mask_data_num

        data_1 = self.get_model_input(x_num, masks=masks)
        x_in_1 = data_1['x_num_t']
        miss_mask_syn_1 = data_1['mask_num']
        model_out_1, _ = self._denoise_fn(x_in_1, data_1['t'], miss_mask_syn_1, train=True)
        loss_gauss_1 = self._gaussian_loss(model_out_1, x_num, data_1['x_num_t'], data_1['t'], data_1['noise'], loss_mask_num)

        data_2 = self.get_model_input(x_num, masks=masks)
        x_in_2 = data_2['x_num_t']
        miss_mask_syn_2 = data_2['mask_num']
        model_out_2, _ = self._denoise_fn(x_in_2, data_2['t'], miss_mask_syn_2, train=True)
        loss_gauss_2 = self._gaussian_loss(model_out_2, x_num, data_2['x_num_t'], data_2['t'], data_2['noise'], loss_mask_num)

        loss_gauss = loss_gauss_1.mean() + loss_gauss_2.mean()

        loss_ssl = torch.zeros((1,)).float()
        if self.ssl_loss_weight > 0:
            loss_ssl = self.ssl_loss(model_out_1, model_out_2) * self.ssl_loss_weight
        return loss_gauss, loss_ssl

    @torch.no_grad()
    def impute(self, x, miss_mask_data):
        x_num = x
        miss_mask_data_num = miss_mask_data[:, :self.num_numerical_features]
        b = x.shape[0]
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)
        z_norm = x_num * (1 - miss_mask_data_num) + z_norm * miss_mask_data_num

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(z_norm.float(), t, miss_mask_data)
            model_out_num = model_out[:, :self.num_numerical_features]
            gaussian_p_sample_output = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)
            z_norm = gaussian_p_sample_output['sample']
            z_norm = x_num * (1 - miss_mask_data_num) + z_norm * miss_mask_data_num

        sample = z_norm.cpu().numpy()

        return sample