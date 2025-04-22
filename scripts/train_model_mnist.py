import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.configLoad import load_config
from models.NCAImpl import NCAModel
from utils.imageUtils import make_seed, make_circle_masks
from dataset.dataloader import cifar10_dataloader, mnist_dataloader


def setup_seed_pool(H, W, N_CHANNEL, POOL_SIZE, device):
    seed = torch.tensor(make_seed((H, W), N_CHANNEL)).permute(2, 0, 1)
    seed[:1] += torch.rand_like(seed[:1]) * 0.05  # add noise only on alpha for MNIST
    pool = seed.unsqueeze(0).repeat(POOL_SIZE, 1, 1, 1).to(device)
    return pool


def compute_loss(x, target):
    mse = F.mse_loss(x[:, 0:1], target)  # Only alpha channel
    return mse, mse  # Simplified


def grey_scale(x):
    imgs = x[:, 0:1].squeeze(1).clamp(0.0, 1.0).cpu().numpy()
    return imgs


def train_step(model, optimizer, scheduler, x, target, steps):
    for _ in range(steps):
        x = model(x)
    loss, mse = compute_loss(x, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return x.detach(), loss.item(), mse.item()


def plot_loss(loss_log, output_path="outputs/loss_curve.png"):
    plt.figure(figsize=(10, 4))
    plt.title("Loss history (log10)")
    plt.plot(np.log10(loss_log), ".", alpha=0.1)
    plt.savefig(output_path)
    plt.close()


def main():
    config = load_config("configs/nca_d_conf.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Hyperparameters ===
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
    SAVE_PATH = config["checkpoint_path"]
    dataset = config["dataset"].lower()

    # === Dataset & Target ===
    loader = (
        mnist_dataloader(batch_size=1)
        if dataset == "mnist"
        else cifar10_dataloader(batch_size=1)
    )
    target_batch, _ = next(iter(loader))
    target_image = target_batch.to(device)
    _, C, H, W = target_image.shape
    print("Target shape:", target_image.shape)

    # === Model & Training Setup ===
    model = NCAModel(N_CHANNEL, HIDDEN_CHANNELS, FIRE_RATE, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    pool = setup_seed_pool(H, W, N_CHANNEL, POOL_SIZE, device)

    # === Training Loop ===
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    loss_log = []

    for i in range(EPOCHS + 1):
        x = pool[:BATCH_SIZE].clone()

        if DAMAGE_N > 0:
            damage = 1.0 - make_circle_masks(DAMAGE_N, H, W).to(device)[..., None]
            x[-DAMAGE_N:] *= damage.permute(0, 3, 1, 2)

        x_out, loss, mse = train_step(
            model,
            optimizer,
            scheduler,
            x,
            target_image.repeat(BATCH_SIZE, 1, 1, 1),
            N_STEPS,
        )

        alpha = x_out[:, 0:1]
        if alpha.max() < 0.1 and i > 100:
            pool[:BATCH_SIZE] = setup_seed_pool(H, W, N_CHANNEL, BATCH_SIZE, device)
        else:
            pool[:BATCH_SIZE] = x_out

        loss_log.append(loss)

        if i % 100 == 0:
            print(
                f"[{i:05d}] Loss: {loss:.6f} | MSE: {mse:.6f} | Alpha Max: {alpha.max():.3f}"
            )

            torch.save(model.state_dict(), SAVE_PATH)

            imgs = grey_scale(x_out[:4])
            grid = np.concatenate(imgs, axis=1)
            plt.imsave(f"outputs/step_{i:05d}.png", grid, cmap="gray")

    # === Loss plot ===
    plot_loss(loss_log)


if __name__ == "__main__":
    main()
