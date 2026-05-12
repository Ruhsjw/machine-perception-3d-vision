import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


def positional_encoding(x, num_frequencies=6, incl_input=True):
    results = []
    if incl_input:
        results.append(x)

    for i in range(num_frequencies):
        frequency = (2.0 ** i) * np.pi
        results.append(torch.sin(frequency * x))
        results.append(torch.cos(frequency * x))

    return torch.cat(results, dim=-1)


class model_2d(nn.Module):
    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        input_dim = (num_frequencies * 2 + 1) * 2
        self.layer_in = nn.Linear(input_dim, filter_size)
        self.act1 = nn.ReLU()
        self.layer = nn.Linear(filter_size, filter_size)
        self.act2 = nn.ReLU()
        self.layer_out = nn.Linear(filter_size, 3)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_in(x)
        x = self.act1(x)
        x = self.layer(x)
        x = self.act2(x)
        x = self.layer_out(x)
        x = self.act_out(x)
        return x


def normalize_coord(height, width, num_frequencies=6):
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x)
    coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    embedded_coordinates = positional_encoding(
        coordinates, num_frequencies=num_frequencies, incl_input=True
    )
    return embedded_coordinates


def train_2d_model(
    test_img,
    num_frequencies,
    device,
    model=model_2d,
    positional_encoding=positional_encoding,
    show=True,
):
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]
    display = 2000

    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)

    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    coordinates = normalize_coord(height, width, num_frequencies=num_frequencies).to(device)
    target = test_img.to(device)

    for i in range(iterations + 1):
        optimizer.zero_grad()

        pred = model2d(coordinates).reshape(height, width, 3)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if i % display == 0 and show:
            psnr = -10.0 * torch.log10(loss)

            print(
                "Iteration %d " % i,
                "Loss: %.4f " % loss.item(),
                "PSNR: %.2f" % psnr.item(),
                "Time: %.2f secs per iter" % ((time.time() - t) / display),
                "%.2f secs in total" % (time.time() - t0),
            )
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(target.detach().cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print("Done!")
    torch.save(model2d.state_dict(), "model_2d_" + str(num_frequencies) + "freq.pt")
    plt.imsave("van_gogh_" + str(num_frequencies) + "freq.png", pred.detach().cpu().numpy())
    return pred.detach().cpu()
