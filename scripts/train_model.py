import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.configLoad import load_config
from models.NCAImpl import NCAModel
from utils.imageUtils import make_seed, make_circle_masks, to_rgb
from dataset.dataloader import cifar10_dataloader, mnist_dataloader


def setup_seed_pool(H, W, N_CHANNEL, POOL_SIZE, device):
    seed = torch.tensor(make_seed((H, W), N_CHANNEL)).permute(2, 0, 1)
    seed[:3] += torch.rand_like(seed[:3]) * 0.05  # léger bruit sur RGB
    pool = seed.unsqueeze(0).repeat(POOL_SIZE, 1, 1, 1).to(device)
    return pool


def compute_loss(x, target):
    mse = F.mse_loss(x[:, : target.shape[1]], target)
    rgb = x[:, :3]
    variance_penalty = -torch.mean(torch.var(rgb, dim=[2, 3]))
    rgb_mean_penalty = ((rgb.mean(dim=(2, 3)) - 0.5) ** 2).mean()  # regularization
    total_loss = mse + 0.1 * variance_penalty + 0.01 * rgb_mean_penalty
    return total_loss, mse, variance_penalty, rgb_mean_penalty


def train_step(model, optimizer, scheduler, x, target, steps):
    for _ in range(steps):
        x = model(x)
    loss, mse, var_penalty, rgb_mean_penalty = compute_loss(x, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return x.detach(), loss.item(), mse.item(), var_penalty.item(), rgb_mean_penalty


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

        x_out, loss, mse, var_penalty, rgb_mean_penalty = train_step(
            model,
            optimizer,
            scheduler,
            x,
            target_image.repeat(BATCH_SIZE, 1, 1, 1),
            N_STEPS,
        )

        # Reset Pool
        rgb = x_out[:, :3]
        if rgb.max() < 0.1 and i > 100:
            # print(f"[{i:05d}] Dead pool detected. Resetting.")
            pool[:BATCH_SIZE] = setup_seed_pool(H, W, N_CHANNEL, BATCH_SIZE, device)
        else:
            pool[:BATCH_SIZE] = x_out

        loss_log.append(loss)
        mean_rgb = [rgb[:, c].mean().item() for c in range(3)]

        if i % 100 == 0:
            print(
                f"[{i:05d}] Loss: {loss:.6f} | MSE: {mse:.6f} | Var Penalty: {var_penalty:.6f} | "
                f"RGB Mean Penalty: {rgb_mean_penalty:.6f} | Mean RGB: {[round(m, 3) for m in mean_rgb]} | Min: {rgb.min():.3f} Max: {rgb.max():.3f}"
            )

            torch.save(model.state_dict(), SAVE_PATH)
            out_rgb = to_rgb(x_out[:4].cpu())
            grid = np.concatenate(out_rgb, axis=1)
            plt.imsave(f"outputs/step_{i:05d}.png", grid)

    # === Loss plot ===
    plot_loss(loss_log)


if __name__ == "__main__":
    main()
