import argparse
import os
from time import localtime, strftime

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.distributions as D
from tqdm import tqdm

import image_dataset
from hmc_vae import HMCVAE
import utils

# fmt: off
parser = argparse.ArgumentParser(description='Configurations parser.')
parser.add_argument('--data_name', type=str, required=True, help='name of the dataset e.g. yacht')  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--data_root', type=str, required=True, help='qualified path to the root of data directory')  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--hidden_channels', type=int, required=True)  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--epochs', type=int, required=True)  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--eval_interval', type=float, default=0, help='interval (in steps) between validation e.g. 5e2')  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--savedir', type=str, default='', help='directory to save logs and checkpoints')  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment you wish to load, this should have been automatically generated')  # pylint: disable=C0301 # noqa: E501
parser.add_argument('--seed', type=int, default=42, help='random seed')  # pylint: disable=C0301 # noqa: E501
# fmt: on

args = parser.parse_args()

if args.experiment_name == "":
    args.experiment_name = (
        f"{args.data_name}_{strftime('%Hh_%Mm_%d_%b_%Y', localtime())}_seed{args.seed}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.eval_interval == 0:  # no evaluation
    tensorboard = None
else:
    tensorboard = SummaryWriter(os.path.join(args.savedir, f"{args.experiment_name}"))
    tensorboard.eval_interval = int(args.eval_interval)
torch.manual_seed(args.seed)

train, test = image_dataset.get(args.data_name, args.data_root)
train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test, batch_size=1000, shuffle=False)

in_channels = train.tensors[0].shape[1]
model = HMCVAE(in_channels, latent_dim=100, hidden_channels=args.hidden_channels, T=10, L=5)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)


def train_variational():
    model.train()
    for x, _ in train_dataloader:
        optimizer.zero_grad()
        x = x.to(device)
        pz = model.encode(x)
        z = pz.rsample()
        x_logits = model.decode(z)
        loss = -model.ELBO(x, x_logits, pz).mean()
        loss.backward()
        optimizer.step()


def train_hmc():
    model.train()
    for x, _ in train_dataloader:
        optimizer.zero_grad()
        x = x.to(device)
        pz = model.encode(x)
        z = pz.rsample()
        with utils.EnableOnly(model, model.encoder.parameters()):
            x_logits = model.decode(z)
            loss = -model.ELBO(x, x_logits, pz).mean()
            loss.backward()
        with utils.EnableOnly(model, model.decoder.parameters()):
            z, accept_prob = model.run_hmc(x, z)
            x_logits = model.decode(z)
            loss = -model.HMC_bound(x, x_logits, z).mean()
            loss.backward()
        optimizer.step()
    print(accept_prob)


def eval_variational():
    model.eval()
    variational = []
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = x.to(device)
            pz = model.encode(x)
            z = pz.rsample()
            x_logits = model.decode(z)
            logp_x = D.Categorical(logits=x_logits).log_prob(x.int()).sum(dim=(-1, -2, -3))
            variational.append(logp_x)
    return torch.cat(variational).mean().item()


def eval_hmc():
    model.eval()
    variational = []
    hmc = []
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = x.to(device)
            pz = model.encode(x)
            z = pz.rsample()
            with utils.EnableOnly(model, model.encoder.parameters()):
                x_logits = model.decode(z)
                logp_x = D.Categorical(logits=x_logits).log_prob(x.int()).sum(dim=(-1, -2, -3))
                variational.append(logp_x)
            with utils.EnableOnly(model, model.decoder.parameters()):
                z, accept_prob = model.run_hmc(x, z)
                x_logits = model.decode(z)
                logp_x = D.Categorical(logits=x_logits).log_prob(x.int()).sum(dim=(-1, -2, -3))
                hmc.append(logp_x)
    return torch.cat(variational).mean().item(), torch.cat(hmc).mean().item()


path = os.path.join(args.savedir, args.experiment_name, "variational")
if os.path.exists(path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    for epoch in tqdm(range(1, args.epochs)):
        train_variational()
        if tensorboard and epoch % args.eval_interval == 0:
            variational = eval_variational()
            tensorboard.add_scalar("variational", variational, epoch)

    os.makedirs(os.path.join(args.savedir, args.experiment_name), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)

for epoch in tqdm(range(1, 11)):
    train_hmc()
    if tensorboard and epoch % args.eval_interval == 0:
        variational, hmc = eval_hmc()
        tensorboard.add_scalar("variational", variational, args.epochs + epoch)
        tensorboard.add_scalar("hmc", hmc, 100 + epoch)

path = os.path.join(args.savedir, args.experiment_name, "hmc")
torch.save({"model_state_dict": model.state_dict()}, path)

if tensorboard:
    tensorboard.flush()
