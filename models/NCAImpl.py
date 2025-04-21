import torch
import torch.nn as nn
import torch.nn.functional as F


class NCAModel(nn.Module):
    def __init__(self, n_channel, hidden_layer, fire_rate, device=None):
        super().__init__()
        self.n_channel = n_channel
        self.hidden_layer = hidden_layer
        self.fire_rate = fire_rate
        self.device = device or torch.device("cpu")

        # Define fixed filters (identity, sobel_x, sobel_y)
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = sobel_x.t()

        # Shape: [3, 1, 3, 3]
        kernel_stack = torch.stack([identity, sobel_x, sobel_y])[:, None, :, :]
        self.register_buffer(
            "kernels", torch.stack([identity, sobel_x, sobel_y])[:, None, :, :]
        )

        # Update network (perception -> hidden -> output)
        self.update = nn.Sequential(
            nn.Conv2d(3 * self.n_channel, self.hidden_layer, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_layer, self.n_channel, kernel_size=1, bias=False),
        )

    def perception(self, x):
        """
        Apply identity, sobel_x, sobel_y to each channel independently using grouped convolution.
        Returns: [B, 3*C, H, W]
        """
        B, C, H, W = x.shape
        filters = self.kernels.to(x.device).repeat(C, 1, 1, 1)  # [3*C, 1, 3, 3]
        x = x.view(B, C, H, W)  # [B, C, H, W]
        out = F.conv2d(x, filters, padding=1, groups=C)  # [B, 3*C, H, W]
        return out

    def get_living_mask(self, x, threshold=0.1):
        """
        Cells are considered alive if alpha channel > threshold.
        """
        alpha = x[:, 0:1, :, :]  # Use channel 0 as activity indicator
        return (
            F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > threshold
        ).float()

    def forward(self, x):
        dx = self.update(self.perception(x))

        if self.training:
            update_mask = (torch.rand_like(x[:, :1]) <= self.fire_rate).float()
            dx = dx * update_mask

        living_mask = self.get_living_mask(x)
        dx = dx * living_mask
        x = x + dx
        x = x * living_mask
        x = torch.clamp(x, 0.0, 1.0)  # ensure all the cells are between 0.0 and 1.0

        # TODO: The Neural Cellular Network are not stable actually

        return x


# TODO add the persistence Logic but that is during the training and test
