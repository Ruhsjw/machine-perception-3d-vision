import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, num_frequencies=6, incl_input=True):
    results = []
    if incl_input:
        results.append(x)

    for i in range(num_frequencies):
        frequency = (2.0 ** i) * np.pi
        results.append(torch.sin(frequency * x))
        results.append(torch.cos(frequency * x))

    return torch.cat(results, dim=-1)


def get_rays(height, width, intrinsics, w_R_c, w_T_c):
    device = intrinsics.device
    dtype = intrinsics.dtype

    f_x = intrinsics[0, 0]
    f_y = intrinsics[1, 1]
    c_x = intrinsics[0, 2]
    c_y = intrinsics[1, 2]

    u, v = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
    )

    x_normalized = (v - c_x) / f_x
    y_normalized = (u - c_y) / f_y

    ray_directions = torch.stack(
        (x_normalized, y_normalized, torch.ones_like(x_normalized)), dim=-1
    )
    ray_directions = ray_directions @ w_R_c.T

    origin = w_T_c.reshape(1, 1, 3).to(device=device, dtype=dtype)
    ray_origins = origin.expand(height, width, 3)

    return ray_origins, ray_directions


def stratified_sampling(ray_origins, ray_directions, near, far, samples):
    height, width = ray_origins.shape[:2]
    step = (far - near) / samples
    depth_points = (
        near
        + torch.arange(
            samples,
            device=ray_origins.device,
            dtype=ray_origins.dtype,
        )
        * step
    ).view(1, 1, samples)
    depth_points = depth_points.expand(height, width, samples)
    ray_points = (
        ray_origins.unsqueeze(-2)
        + ray_directions.unsqueeze(-2) * depth_points.unsqueeze(-1)
    )
    return ray_points, depth_points


class nerf_model(nn.Module):
    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        x_dim = (num_x_frequencies * 2 + 1) * 3
        d_dim = (num_d_frequencies * 2 + 1) * 3

        self.layers = nn.ModuleDict(
            {
                "layer_1": nn.Linear(x_dim, filter_size),
                "layer_2": nn.Linear(filter_size, filter_size),
                "layer_3": nn.Linear(filter_size, filter_size),
                "layer_4": nn.Linear(filter_size, filter_size),
                "layer_5": nn.Linear(filter_size + x_dim, filter_size),
                "layer_6": nn.Linear(filter_size, filter_size),
                "layer_7": nn.Linear(filter_size, filter_size),
                "layer_8": nn.Linear(filter_size, filter_size),
                "layer_s": nn.Linear(filter_size, 1),
                "layer_9": nn.Linear(filter_size, filter_size),
                "layer_10": nn.Linear(filter_size + d_dim, 128),
                "layer_11": nn.Linear(128, 3),
            }
        )

        nn.init.constant_(self.layers["layer_s"].bias, 0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d):
        x_input = x

        x = F.relu(self.layers["layer_1"](x))
        x = F.relu(self.layers["layer_2"](x))
        x = F.relu(self.layers["layer_3"](x))
        x = F.relu(self.layers["layer_4"](x))
        x = torch.cat((x, x_input), dim=-1)
        x = F.relu(self.layers["layer_5"](x))
        x = F.relu(self.layers["layer_6"](x))
        x = F.relu(self.layers["layer_7"](x))
        x = F.relu(self.layers["layer_8"](x))

        sigma = F.relu(self.layers["layer_s"](x)).squeeze(-1)

        features = self.layers["layer_9"](x)
        rgb_input = torch.cat((features, d), dim=-1)
        rgb = F.relu(self.layers["layer_10"](rgb_input))
        rgb = self.sigmoid(self.layers["layer_11"](rgb))

        return rgb, sigma


def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    def get_chunks(inputs, chunksize=2**15):
        return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    _, _, samples, _ = ray_points.shape

    ray_directions = ray_directions / torch.norm(
        ray_directions, dim=-1, keepdim=True
    ).clamp_min(1e-9)
    ray_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples, -1)

    ray_points = positional_encoding(
        ray_points.reshape(-1, 3),
        num_frequencies=num_x_frequencies,
        incl_input=True,
    )
    ray_directions = positional_encoding(
        ray_directions.reshape(-1, 3),
        num_frequencies=num_d_frequencies,
        incl_input=True,
    )

    ray_points_batches = get_chunks(ray_points)
    ray_directions_batches = get_chunks(ray_directions)

    return ray_points_batches, ray_directions_batches


def volumetric_rendering(rgb, s, depth_points):
    device = s.device
    dtype = s.dtype
    height, width, samples = s.shape

    s = F.relu(s)
    delta = torch.zeros((height, width, samples), device=device, dtype=dtype)
    delta[:, :, :-1] = depth_points[:, :, 1:] - depth_points[:, :, :-1]
    delta[:, :, -1] = 1e9

    alpha = 1.0 - torch.exp(-s * delta)
    transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones((height, width, 1), device=device, dtype=dtype),
                1.0 - alpha + 1e-10,
            ],
            dim=-1,
        ),
        dim=-1,
    )[:, :, :-1]

    weights = transmittance * alpha
    rec_image = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    rec_image = rec_image.clamp(0.0, 1.0)

    return rec_image


def one_forward_pass(
    height,
    width,
    intrinsics,
    pose,
    near,
    far,
    samples,
    model,
    num_x_frequencies,
    num_d_frequencies,
):
    ray_origins, ray_directions = get_rays(
        height, width, intrinsics, pose[:3, :3], pose[:3, 3]
    )

    ray_points, depth_points = stratified_sampling(
        ray_origins, ray_directions, near, far, samples
    )

    ray_points_batches, ray_directions_batches = get_batches(
        ray_points, ray_directions, num_x_frequencies, num_d_frequencies
    )

    rgb = None
    sigma = None
    for i in range(len(ray_points_batches)):
        rgb_batch, sigma_batch = model(ray_points_batches[i], ray_directions_batches[i])
        if rgb is None:
            rgb = rgb_batch
            sigma = sigma_batch
        else:
            rgb = torch.cat((rgb, rgb_batch), dim=0)
            sigma = torch.cat((sigma, sigma_batch), dim=0)

    rgb = rgb.reshape(height, width, samples, 3)
    sigma = sigma.reshape(height, width, samples)
    rec_image = volumetric_rendering(rgb, sigma, depth_points)

    return rec_image
