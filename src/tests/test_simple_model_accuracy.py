import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from pcdvq.codebooks import PCDVQCodebook
from pcdvq.utils import default_device, random_seed
from pcdvq.quantizer import Quantizer, quantize_linear_inplace


def calc_acc(mdl, dl):
    "calculates accuracy"
    mdl.eval()
    dev, corr, n = next(mdl.parameters()).device, 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            corr += (mdl(xb).argmax(dim=1) == yb).sum().item()
            n += len(xb)

    return corr / n


if __name__ == "__main__":
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)

    random_seed(42)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=6, persistent_workers=True)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    model.to(default_device)
    opt = optim.SGD(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    n_epoch = 5
    for epoch in tqdm(range(n_epoch)):
        for xb, yb in tqdm(dl):
            xb, yb = xb.to(default_device), yb.to(default_device)
            opt.zero_grad()
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    accuracies = {}
    accuracies["original"] = calc_acc(model, dl)
    print(f'Initial Accuracy: {accuracies["original"] }')

    original_state = copy.deepcopy(model.state_dict())

    codebooks = [
        "e8p_8_2.pt",
        "e8_8_2.pt",
        "e8p_10_2.pt",
        "e8_10_2.pt",
        "e8p_12_2.pt",
        "e8_12_2.pt",
        "e8p_14_2.pt",
        "e8_14_2.pt",
        "e8p_16_2.pt",
        "e8_16_2.pt",
    ]
    path = "./codebooks/codebook_"

    for codebook_name in codebooks:
        codebook = PCDVQCodebook()
        codebook.load(f"{path}{codebook_name}")
        quantizer = Quantizer(codebook)
        quantize_linear_inplace(model, quantizer)
        accuracies[codebook_name] = calc_acc(model, dl)
        model.load_state_dict(original_state)

    print("\n--- Accuracy Results ---")
    print(f"{'Codebook':<20} | {'Accuracy':<10}")
    print("-" * 33)
    for codebook, acc in accuracies.items():
        print(f"{codebook:<20} | {acc:<10.4f}")
    print("-" * 33)
