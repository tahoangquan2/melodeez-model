import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils import data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Parameter
import math
import visdom
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import faiss
import torchvision.models as models

class Visualizer:
    def __init__(self, env='default', **kwargs):
        try:
            import visdom
            self.vis = visdom.Visdom(env=env, **kwargs)
            self.vis.close()
            self.iters = {}
            self.lines = {}
            self.visdom_available = True
        except ImportError:
            print("Visdom not available.")
            self.visdom_available = False

    def display_current_results(self, iters, x, name='train_loss'):
        if not self.visdom_available:
            return
        iters_list = self.iters.setdefault(name, [])
        lines_list = self.lines.setdefault(name, [])

        iters_list.append(iters)
        lines_list.append(x)

        try:
            self.vis.line(X=np.array(iters_list),
                          Y=np.array(lines_list),
                          win=name,
                          opts=dict(legend=[name], title=name))
        except Exception as e:
            print(f"Visualization failed: {e}")

class SEBlock(nn.Module):
    def __init__(self, channel):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16),
            nn.ReLU(),
            nn.Linear(channel // 16, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CustomResNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(CustomResNet, self).__init__()
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.se = SEBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return F.normalize(x, p=2, dim=1)

class AudioDataset(Dataset):
    def __init__(self, root_dir, list_file, input_shape=(1, 80, 630)):
        self.root_dir = root_dir
        self.input_shape = (input_shape[1], input_shape[2])

        with open(list_file, 'r') as f:
            self.samples = [(os.path.join(root_dir, line.split()[0]),
                           int(line.split()[1])) for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path)
        if data.shape[1] >= self.input_shape[1]:
            data = data[:, :self.input_shape[1]]
        else:
            result = np.zeros(self.input_shape, dtype=np.float32)
            result[:, :data.shape[1]] = data
            data = result
        return torch.from_numpy(data).float().unsqueeze(0), label

def read_val(path_val, data_root):
    dict_data = []
    with open(path_val, 'r') as files:
        for line in files.read().splitlines():
            typ = 'song' if 'song' in line else 'hum'
            filepath, lbl = line.split(' ')
            dict_data.append({
                'path': os.path.join(data_root, filepath),
                'label': lbl,
                'type': typ
            })
    return dict_data

def calculate_mrr(model, data_val, input_shape):
    model.eval()
    device = next(model.parameters()).device
    index = faiss.IndexFlatL2(512)

    song_features, song_labels = [], []
    hum_features, hum_labels = [], []

    with torch.no_grad():
        for item in data_val:
            data = np.load(item['path'])
            if data.shape[1] >= input_shape[1]:
                data = data[:, :input_shape[1]]
            else:
                result = np.zeros(input_shape, dtype=np.float32)
                result[:, :data.shape[1]] = data
                data = result

            tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
            feature = model(tensor).cpu().numpy()

            if item['type'] == 'song':
                song_features.append(feature)
                song_labels.append(item['label'])
            else:
                hum_features.append(feature)
                hum_labels.append(item['label'])

    if not song_features or not hum_features:
        return 0.0

    index.add(np.vstack(song_features))
    hum_features = np.vstack(hum_features)

    mrr_sum = 0
    distances, indices = index.search(hum_features, 10)

    for i, label in enumerate(hum_labels):
        for rank, idx in enumerate(indices[i]):
            if song_labels[idx] == label:
                mrr_sum += 1.0 / (rank + 1)
                break

    return mrr_sum / len(hum_labels)

def train_model_3(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AudioDataset(opt.train_root, opt.train_list, opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)

    print(f'{len(trainloader)} train iters per epoch')

    model = CustomResNet(feature_dim=512)
    model = model.to(device)
    model = DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    mrr_best = 0
    patience = 5
    patience_counter = 0

    for epoch in range(opt.max_epoch):
        model.train()
        total_loss = 0

        try:
            for i, (data, label) in enumerate(trainloader):
                data = data.to(device)
                label = label.to(device)

                feature = model(data)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(feature, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % opt.print_freq == 0:
                    print(f'Epoch: [{epoch}][{i}/{len(trainloader)}]\tLoss: {loss.item():.4f}')

            avg_loss = total_loss / len(trainloader)
            print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')

            if (epoch + 1) % 5 == 0:
                data_val = read_val(opt.val_list, opt.train_root)
                mrr = calculate_mrr(model, data_val, opt.input_shape)
                print(f'Validation MRR = {mrr:.4f}')

                if mrr > mrr_best:
                    mrr_best = mrr
                    torch.save(model.state_dict(), os.path.join(opt.checkpoints_path, 'best_model.pth'))
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break

            scheduler.step()

        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            continue

    print(f"Training completed. Best MRR: {mrr_best:.4f}")
    return model
