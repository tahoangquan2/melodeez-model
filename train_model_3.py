import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
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
    def __init__(self, feature_dim=512, backbone='resnet18'):
        super(CustomResNet, self).__init__()
        if backbone == 'resnet18':
            model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

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
    def __init__(self, root_dir, list_file, input_shape):
        self.root_dir = root_dir
        if isinstance(input_shape, tuple):
            self.input_shape = (input_shape[1], input_shape[2])
        else:
            raise ValueError("input_shape must be a tuple of (channels, height, width)")

        with open(list_file, 'r') as f:
            self.samples = [(os.path.join(root_dir, line.split()[0]),
                           int(line.split()[1])) for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path)
        label = label - 1
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
            filepath, label = line.split(' ')
            typ = 'song' if 'song' in filepath else 'hum'
            dict_data.append({
                'path': os.path.join(data_root, filepath),
                'type': typ,
                'id': int(label)
            })
    return dict_data

def calculate_mrr(model, data_val, input_shape):
    model.eval()
    device = next(model.parameters()).device

    feature_dim = model.module.fc.out_features if hasattr(model, 'module') else model.fc.out_features
    index = faiss.IndexFlatL2(feature_dim)

    song_features = []
    song_ids = []
    hum_features = []
    hum_ids = []

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
                song_ids.append(item['id'])
            else:
                hum_features.append(feature)
                hum_ids.append(item['id'])

    if not song_features or not hum_features:
        return 0.0

    index.add(np.vstack(song_features))
    hum_features = np.vstack(hum_features)

    mrr_sum = 0
    distances, indices = index.search(hum_features, 10)

    for i, query_id in enumerate(hum_ids):
        for rank, idx in enumerate(indices[i]):
            if song_ids[idx] == query_id:
                mrr_sum += 1.0 / (rank + 1)
                break

    mrr = mrr_sum / len(hum_ids)

    print(f"\nMRR Evaluation Statistics:")
    print(f"Total unique queries: {len(set(hum_ids))}")
    print(f"Total unique songs: {len(set(song_ids))}")
    print(f"Total queries: {len(hum_ids)}")
    print(f"Total songs in index: {len(song_ids)}")
    print(f"Final MRR: {mrr:.4f}")

    return mrr
