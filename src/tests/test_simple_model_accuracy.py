import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.pcdvq.codebooks import PCDVQCodebook
from src.pcdvq.quantizer import Quantizer, quantize_linear_inplace


def calc_acc(mdl, dl):
    'calculates accuracy'
    mdl.eval()
    dev,corr,n = next(mdl.parameters()).device, 0., 0
    with torch.no_grad():
        for xb,yb in dl:
            xb,yb = xb.to(dev),yb.to(dev)
            corr += (mdl(xb).argmax(dim=1)==yb).sum().item()
            n += len(xb)

    return corr/n

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST('./data', train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=6, persistent_workers=True)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(dev)
opt = optim.SGD(model.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()

for epoch in tqdm(range(5)):
    for xb,yb in tqdm(dl):
        xb,yb = xb.to(dev),yb.to(dev)
        opt.zero_grad()
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()

    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print(f'Initial Accuracy: {calc_acc(model, dl)}')

original_state = copy.deepcopy(model.state_dict())

codebooks = ["e8p_8_2.pt", "e8_8_2.pt", "e8p_10_2.pt", "e8_10_2.pt", "e8p_12_2.pt", "e8_12_2.pt", "e8p_14_2.pt", "e8_14_2.pt", "e8p_16_2.pt", "e8_16_2.pt"]
path = '/home/lama/PCDVQ/codebooks/codebook_'

for codebook_name in codebooks:
    codebook = PCDVQCodebook()
    codebook.load(f'{path}{codebook_name}')
    quantizer = Quantizer(codebook)
    quantize_linear_inplace(model, quantizer)
    print(f'Accuracy with {codebook_name}: {calc_acc(model, dl)}')
    model.load_state_dict(original_state)
