import torch
import numpy as np
import zero
from model import SimpDM
from load_data import make_dataset, prepare_fast_dataloader
from model.modules import MLPDiffusion

import pandas as pd
import argparse
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.utils.metrics import generate_score

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, epochs, device=torch.device('cuda:0'), data=None):
        self.diffusion = diffusion
        self.train_iter = train_iter
        self.epochs = epochs
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.log_every = 100
        self.print_every = 1000
        self.ema_every = 1000
        self.data = data

    def _anneal_lr(self, step):
        frac_done = step / self.epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, mask):
        x = x.to(self.device)
        mask = mask.to(self.device)

        self.optimizer.zero_grad()
        loss_gauss, loss_ssl = self.diffusion.train_iter(x, mask)
        loss = loss_gauss + loss_ssl
        loss.backward()
        self.optimizer.step()

        return loss_gauss, loss_ssl

    def run_loop(self):
        step = 0
        curr_loss_gauss = 0.0
        curr_loss_ssl = 0.0

        curr_count = 0
        while step < self.epochs:
            x, mask = next(self.train_iter)
            batch_loss_gauss, batch_loss_ssl = self._run_step(x, mask)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)
            curr_loss_ssl += batch_loss_ssl.item() * len(x)

            if (step + 1) % self.log_every == 0:
                gloss = curr_loss_gauss / curr_count
                ssl_loss = curr_loss_ssl / curr_count
                if (step + 1) % self.print_every == 0:
                    print('Step {}/{}  DM Loss: {:.6f} SSL Loss:{:.6f}, Sum: {:.6f}'
                          .format((step + 1), self.epochs, gloss, ssl_loss, gloss + ssl_loss))

                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_ssl = 0.0

            step += 1

def summarize_results(results, args):
    final_result = {}
    all_result = {}
    for key in results[0]:
        rmses = []
        for trial in range(args.n_trial):
            rmses.append(results[trial][key])
        rmse_mean, rmse_std = generate_score(rmses)
        final_result[key] = '{:.4f}+-{:.4f}'.format(rmse_mean, rmse_std)
        all_result[key] = rmses
        print('{}: {}'.format(key, final_result[key]))

def main(args, device = torch.device('cuda:0'), seed = 0):

    ####################### LOAD DATA #######################
    zero.improve_reproducibility(seed)
    D = make_dataset(args)
    num_numerical_features = D.X_num['x_miss'].shape[1]
    d_in = num_numerical_features
    d_out = num_numerical_features

    model = MLPDiffusion(d_in=d_in, d_out=d_out, d_layers=[args.hidden_units] * args.num_layers)
    model.to(device)


    ####################### TRAIN #######################
    train_loader = prepare_fast_dataloader(D, split='train', batch_size=args.batch_size)

    diffusion = SimpDM(num_numerical_features=num_numerical_features, denoise_fn=model, device=device,
                       num_timesteps=args.num_timesteps, gammas=args.gammas, ssl_loss_weight=args.ssl_loss_weight)
    diffusion.to(device)
    diffusion.train()
    trainer = Trainer(diffusion, train_loader, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
                      device=device, data=D)
    trainer.run_loop()

    ####################### IMPUTE #######################
    diffusion.eval()

    X = torch.from_numpy(D.X_num['x_miss']).float()
    X = torch.nan_to_num(X, nan=-1)
    mask = torch.from_numpy(D.X_num['miss_mask']).float()

    x_imputed = diffusion.impute(X.to(device), mask.to(device))

    ####################### EVALUATE #######################
    result = {}

    x_test_gt = D.X_num['x_gt']
    mask = D.X_num['miss_mask']

    rmse = RMSE(x_imputed, x_test_gt, mask)
    result['rmse'] = rmse
    print(rmse)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # exp param
    parser.add_argument("--n_trial", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")

    # data param
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["iris", "yacht", "housing", "diabetes", "blood", "energy", "german", "concrete", "yeast",
                                "airfoil", "wine_red", "abalone", "wine_white", "phoneme", "power", "ecommerce", "california"])
    parser.add_argument("--scenario", type=str, default="MCAR")
    parser.add_argument("--missing_ratio", type=float, default=0.3)

    # training params
    parser.add_argument("--epochs", type=int, default=10000) # 10000
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4096)

    # model params
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--ssl_loss_weight", type=float, default=1)
    parser.add_argument("--gammas", type=str, default="1_0.8_0.001")

    args = parser.parse_args()
    device = torch.device(args.device)

    args.gammas = args.gammas.split('_')
    args.gammas = [float(gamma) for gamma in args.gammas]

    timer = zero.Timer()
    timer.run()

    results = []
    for trial in range(args.n_trial):
        result = main(seed=trial, device=device, args=args)
        results.append(result)
    summarize_results(results, args)