import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
import faiss
from logger import logger

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class IRBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(IRBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class ResNetFace(nn.Module):
    def __init__(self, feature_dim=512):
        super(ResNetFace, self).__init__()

        # Layer configuration
        layers = [2, 3, 4, 3]

        # Initial layers with stride
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual layers with specified channel configurations
        self.layer1 = self._make_layer(IRBlock, 64, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(IRBlock, 128, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(IRBlock, 256, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(IRBlock, 512, 128, layers[3], stride=2)

        # Final layers
        self.bn4 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(12800, feature_dim)
        self.bn5 = nn.BatchNorm1d(feature_dim)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.bn5(x)

        return x

class AudioDataset(Dataset):
    def __init__(self, root_dir, list_file, input_shape):
        self.root_dir = root_dir
        if isinstance(input_shape, tuple):
            self.input_shape = (input_shape[1], input_shape[2])
        else:
            raise ValueError("input_shape must be a tuple of (channels, height, width)")

        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.samples = []
            for line in lines:
                try:
                    parts = line.rstrip('\n').rsplit(' ', 1)
                    if len(parts) != 2:
                        logger.warning(f"Malformed line: {line}")
                        continue

                    path, label_str = parts
                    path = path.replace('\\', '/')
                    label = int(label_str)

                    full_path = os.path.join(root_dir, path)
                    if not os.path.exists(full_path):
                        logger.warning(f"File not found: {full_path}")
                        continue

                    self.samples.append((full_path, label))
                except Exception as e:
                    logger.warning(f"Error processing line: {line}, Error: {e}")

        if not self.samples:
            raise RuntimeError("No valid samples found in the dataset")

        logger.info(f"Loaded {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            npy_path, label = self.samples[idx]
            data = np.load(npy_path)

            if len(data.shape) != 2:
                raise ValueError(f"Expected 2D array, got shape {data.shape}")

            if data.shape[1] >= self.input_shape[1]:
                data = data[:, :self.input_shape[1]]
            else:
                result = np.zeros(self.input_shape, dtype=np.float32)
                result[:, :data.shape[1]] = data
                data = result

            if label <= 0:
                raise ValueError(f"Invalid label value: {label}")

            label = label - 1

            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Data contains NaN or Inf values")

            return torch.from_numpy(data).float().unsqueeze(0), label

        except Exception as e:
            logger.error(f"Error loading sample {idx} from {npy_path}: {e}")
            return torch.zeros((1,) + self.input_shape), 0

def read_val(path_val, data_root):
    dict_data = []
    with open(path_val, 'r', encoding='utf-8') as files:
        for line in files:
            parts = line.rstrip('\n').rsplit(' ', 1)
            if len(parts) != 2:
                print(f"Warning: Malformed validation line: {line}")
                continue
            filepath, label = parts
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
