import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils import data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.nn import Parameter
import math
import visdom
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import faiss

# Utilities
@torch.no_grad()
def load_image(npy_path, input_shape=(80, 630)):
    data = np.load(npy_path)
    # Preallocate directly and copy if shorter
    if data.shape[1] >= input_shape[1]:
        data = data[:, :input_shape[1]]
    else:
        # Pad the time dimension
        result = np.zeros(input_shape, dtype=np.float32)
        result[:, :data.shape[1]] = data
        data = result

    # Directly convert to torch tensor
    return torch.from_numpy(data).float().unsqueeze(0)

@torch.no_grad()
def get_feature(model, image):
    """Extract features from image using the given model."""
    if image.dim() == 2:
        # Add batch and channel dims
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        # Add batch dim only
        image = image.unsqueeze(0)

    device = torch.device("cuda")
    image = image.to(device, non_blocking=True)
    with torch.no_grad():
        output = model(image)
    # Normalize using PyTorch (L2 norm)
    output = F.normalize(output, p=2, dim=1)
    # Move to CPU and convert to numpy
    output = output.cpu().numpy().flatten()
    return np.expand_dims(output, axis=0)

# Visualizer
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

# Loss Function
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# Metric Learning Functions
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.clamp(cosine.pow(2), max=1.0)))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# ResNet Model
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        super(ResNetFace, self).__init__()
        self.inplanes = 64
        self.use_se = use_se

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool21 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1])
        self.layer3 = self._make_layer(block, 512, layers[2])
        self.layer4 = self._make_layer(block, 128, layers[3])

        self.bn4 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout()

        # Flatten size: 128 * 2 * 19 = 4864
        self.fc5 = nn.Linear(4864, 512)
        self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)          # (B,64,80,630)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool21(x)      # (B,64,40,315)

        x = self.layer1(x)         # (B,128,40,315)
        x = self.maxpool21(x)      # (B,128,20,157)
        x = self.layer2(x)         # (B,256,20,157)
        x = self.maxpool21(x)      # (B,256,10,78)
        x = self.layer3(x)         # (B,512,10,78)
        x = self.maxpool21(x)      # (B,512,5,39)
        x = self.layer4(x)         # (B,128,5,39)
        x = self.maxpool21(x)      # (B,128,2,19)

        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        # Only normalize if batch size > 1
        if x.size(0) > 1:
            x = self.bn5(x)
        return x

# Dataset
class AudioDataset(data.Dataset):
    def __init__(self, root_dir, list_file, phase='train', input_shape=(1, 80, 630),
                 mp3aug_ratio=0.5, npy_aug=True):
        self.root_dir = root_dir
        self.phase = phase
        # For load_image we only need the mel and time dimensions
        self.input_shape = (input_shape[1], input_shape[2])

        with open(list_file, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            path, label = line.strip().split()
            self.samples.append((os.path.join(root_dir, path), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        # load_image will return tensor of shape (1, 80, 630)
        spectrogram = load_image(npy_path, self.input_shape)
        return spectrogram, label

# Validation functions
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

def mean_reciprocal_rank(preds, gt, k=10):
    k = min(k, len(preds))
    for rank in range(k):
        if preds[rank] == gt:
            return 1.0 / (rank + 1)
    return 0.0

@torch.no_grad()
def mrr_score(model, dict_data, input_shape):
    index = faiss.IndexFlatL2(512)
    index2id = {}
    id_counter = 0
    result_search = []
    model.eval()

    # Index songs
    song_features = []
    song_labels = []
    for item in dict_data:
        if item['type'] == 'song':
            image = load_image(item['path'], input_shape[1:])
            feature = get_feature(model, image)
            song_features.append(feature)
            song_labels.append(item['label'])

    # Bulk add for efficiency
    if song_features:
        index.add(np.concatenate(song_features, axis=0))
        for i, lbl in enumerate(song_labels):
            index2id[str(i)] = lbl

    # Query hum files
    hum_features = []
    hum_labels = []
    for item in dict_data:
        if item['type'] == 'hum':
            image = load_image(item['path'], input_shape[1:])
            feature = get_feature(model, image)
            hum_features.append(feature)
            hum_labels.append(item['label'])

    # Search all hum queries at once if memory allows
    if hum_features:
        hum_features_batch = np.concatenate(hum_features, axis=0)
        distances, indices = index.search(hum_features_batch, 10)
        for i, lbl in enumerate(hum_labels):
            preds = [index2id[str(idx)] for idx in indices[i]]
            result_search.append((lbl, preds))

    mrr = sum(mean_reciprocal_rank(preds, lbl) for lbl, preds in result_search) / len(result_search) if result_search else 0.0
    return mrr

# Training function
def train_model_3(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    train_dataset = AudioDataset(opt.train_root, opt.train_list,
                                 phase='train',
                                 input_shape=opt.input_shape,
                                 mp3aug_ratio=opt.mp3aug_ratio,
                                 npy_aug=opt.npy_aug)

    # Use pinned_memory for faster data transfer to GPU
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)

    print(f'{len(trainloader)} train iters per epoch')

    criterion = FocalLoss(gamma=2) if opt.loss == 'focal_loss' else nn.CrossEntropyLoss()
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=opt.use_se)
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)

    model = model.to(device)
    model = DataParallel(model)
    metric_fc = metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': metric_fc.parameters()}],
                                lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
