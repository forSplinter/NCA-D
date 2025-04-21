import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils.configLoad import load_config
from models.NCAImpl import NCAModel
from utils.imageUtils import make_seed, make_circle_masks, to_rgb
from dataset.dataloader import cifar10_dataloader, mnist_dataloader


def plot_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.title("Loss history (log10)")
    plt.plot(np.log10(loss_log), ".", alpha=0.1)
    plt.show()


# === Load config and device ===
config = load_config("configs/nca_d_conf.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters from config ===
N_CHANNEL = config["n_channel"]
HIDDEN_CHANNELS = config["hidden_channels"]
FIRE_RATE = config["fire_rate"]
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
POOL_SIZE = config["pool_size"]
N_STEPS = config["n_steps"]
LR = config["lr"]
LR_GAMMA = config["lr_gamma"]
DAMAGE_N = config["damage_n"]
USE_PATTERN_POOL = config["use_pattern_pool"]
SAVE_PATH = config["checkpoint_path"]

# === Load dataset ===
dataset = config["dataset"].lower()
if dataset == "mnist":
    loader = mnist_dataloader(batch_size=1, train=True)
elif dataset == "cifar10":
    loader = cifar10_dataloader(batch_size=1, train=True)
else:
    raise ValueError(f"Unknown dataset: {dataset}")

target_batch, _ = next(iter(loader))
target_image = target_batch.to(device)  # [1, C, H, W]
_, C, H, W = target_image.shape

# === Model, optimizer, seed and pool ===
model = NCAModel(N_CHANNEL, HIDDEN_CHANNELS, FIRE_RATE, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

seed = torch.tensor(make_seed((H, W), N_CHANNEL)).permute(2, 0, 1)
pool = torch.repeat_interleave(seed.unsqueeze(0), POOL_SIZE, dim=0).clone().to(device)


def mse_loss(x, target):
    return F.mse_loss(x[:, : target.shape[1]], target)


def train_step(x, target, steps):
    for _ in range(steps):
        x = model(x)
    loss = mse_loss(x, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x.detach(), loss.item()


# === Training loop ===
loss_log = []
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

for i in range(EPOCHS + 1):
    x = pool[:BATCH_SIZE].clone()

    if DAMAGE_N > 0:
        # TODO: Actually the model is good for stabilise he tend to a local mimun with epochs=100 but it's not enough for the model to learn we need to add some damage
        # TODO: maybe change the learn steps in the paper their change between [64-96]
        damage = (
            1.0 - make_circle_masks(DAMAGE_N, H, W).to(device)[..., None]
        )  # TODO: the function don't return Tensor need to change that
        x[-DAMAGE_N:] *= damage.permute(0, 3, 1, 2)

    x_out, loss = train_step(x, target_image.repeat(BATCH_SIZE, 1, 1, 1), N_STEPS)
    pool[:BATCH_SIZE] = x_out
    loss_log.append(loss)

    if i % 100 == 0:
        print(f"[{i:05d}] Loss: {loss:.6f}")
        torch.save(model.state_dict(), SAVE_PATH)

        out_rgb = to_rgb((x_out[:4].cpu()))
        grid = grid = np.concatenate(out_rgb, axis=1)
        plt.imsave(f"outputs/step_{i:05d}.png", grid)

# === Final loss plot ===
plt.figure()
plt.plot(loss_log)
plt.title("Loss over training")
plt.xlabel("Step")
plt.ylabel("MSE")
plt.savefig("outputs/loss_curve.png")
