import torch.optim as optim

import settings
from loader import get_train_loaders
from unet_models import UNetResNet
from lovasz_losses import lovasz_hinge

def train():

    model = UNetResNet(50, 2, pretrained=True).cuda()
    criterion = lovasz_hinge 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader, val_loader = get_train_loaders(2)

    for batch_idx, data in enumerate(train_loader):
        img, mask = data
        img, mask = img.cuda(), mask.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}\r')

if __name__ == '__main__':
    train()

