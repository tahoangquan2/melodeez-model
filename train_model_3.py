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

# Utilities
def load_image(npy_path, input_shape=(80, 630)):
    """Load and preprocess spectrogram
    Args:
        npy_path: Path to .npy file
        input_shape: Expected shape (n_mels, time_steps)
    """
    data = np.load(npy_path)
    n_mels, time_steps = data.shape

    result = np.zeros(input_shape)

    # Handle time dimension (truncate or pad)
    if time_steps >= input_shape[1]:
        result = data[:, :input_shape[1]]
    else:
        result[:, :time_steps] = data

    result = torch.from_numpy(result).float().unsqueeze(0)
    return result

def get_feature(model, image):
    """Extract features from image
    Args:
        model: trained model
        image: tensor of shape (1, H, W) or (H, W)
    Returns:
        numpy matrix of features
    """
    # Ensure 4D input: (batch, channel, height, width)
    if image.dim() == 2:
        # Add batch and channel dims
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        # Add batch dim only
        image = image.unsqueeze(0)

    data = image.to(torch.device("cuda"))

    with torch.no_grad():
        output = model(data)

    output = output.cpu().detach().numpy()
    output = normalize(output).flatten()
    return np.matrix(output)

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
            print("Visdom not available. Training will continue without visualization.")
            self.visdom_available = False

    def display_current_results(self, iters, x, name='train_loss'):
        if not self.visdom_available:
            return

        if name not in self.iters:
            self.iters[name] = []
        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)

        try:
            self.vis.line(X=np.array(self.iters[name]),
                         Y=np.array(self.lines[name]),
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
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
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
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()

        # Input shape: (batch, 1, 80, 630)
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

        # Calculate feature size:
        # Input: 80x630
        # After first maxpool: 40x315
        # After 4 more maxpools: 2x19
        # With 128 channels: 128 * 2 * 19 = 4864
        self.fc5 = nn.Linear(4864, 512)
        self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)          # (B, 64, 80, 630)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool21(x)      # (B, 64, 40, 315)

        x = self.layer1(x)         # (B, 128, 40, 315)
        x = self.maxpool21(x)      # (B, 128, 20, 157)
        x = self.layer2(x)         # (B, 256, 20, 157)
        x = self.maxpool21(x)      # (B, 256, 10, 78)
        x = self.layer3(x)         # (B, 512, 10, 78)
        x = self.maxpool21(x)      # (B, 512, 5, 39)
        x = self.layer4(x)         # (B, 128, 5, 39)
        x = self.maxpool21(x)      # (B, 128, 2, 19)

        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten: (B, 128 * 2 * 19)
        x = self.fc5(x)            # (B, 512)
        if x.shape[0] > 1:
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
    files = open(path_val, 'r+')
    dict_data = []
    for line in files.read().splitlines():
        if 'song' in line:
            type = 'song'
        else:
            type = 'hum'
        dict_data.append({
            'path': os.path.join(data_root, line.split(' ')[0]),
            'label': line.split(' ')[1],
            'type': type
        })
    files.close()
    return dict_data

def mean_reciprocal_rank(preds, gt: str, k: int=10):
    preds = preds[: min(k, len(preds))]
    score = 0
    for rank, pred in enumerate(preds):
        if pred == gt:
            score = 1 / (rank + 1)
            break
    return score

def mrr_score(model, dict_data, input_shape):
    import faiss
    index = faiss.IndexFlatL2(512)
    index2id = {"-1": ""}
    id = 0
    count = 0
    s_0 = 0
    s_1 = 0
    result_search = []
    # Ensure model is in eval mode
    model.eval()

    # Index all songs
    print("Indexing songs...")
    for item in dict_data:
        if item['type'] == 'song':
            path_song = item['path']
            image = load_image(path_song, input_shape[1:])
            feature = get_feature(model, image)
            index.add(feature)
            index2id[str(id)] = item['label']
            id += 1

    # Search with queries
    print("Processing queries...")
    for item in dict_data:
        if item['type'] == 'hum':
            path_hum = item['path']
            image = load_image(path_hum, input_shape[1:])
            feature = get_feature(model, image)
            distances, indices = index.search(feature, 10)

            preds = [str(index2id[str(idx)]) for idx in indices[0]]
            result_search.append([item['label'], preds])

            mrr = mean_reciprocal_rank(preds, item['label'])
            if mrr == 1.0:
                s_0 += distances[0, 0]
                s_1 += distances[0, 1]
                count += 1

    mrr = sum(mean_reciprocal_rank(row[1], row[0]) for row in result_search) / len(result_search)
    if count > 0:
        print(f"Perfect matches: {count}")
        print(f"Average distance (top-1): {s_0 / count:.4f}")
        print(f"Average distance (top-2): {s_1 / count:.4f}")
    return mrr

# Training function
def train_model_3(opt):
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    # Initialize dataset and dataloader
    train_dataset = AudioDataset(opt.train_root, opt.train_list, phase='train',
                                input_shape=opt.input_shape,
                                mp3aug_ratio=opt.mp3aug_ratio, npy_aug=opt.npy_aug)

    trainloader = data.DataLoader(train_dataset,
                                batch_size=opt.train_batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers)

    print(f'{len(trainloader)} train iters per epoch')

    # Initialize components
    criterion = FocalLoss(gamma=2) if opt.loss == 'focal_loss' else nn.CrossEntropyLoss()
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=opt.use_se)
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)

    # Move to GPU
    model = model.to(device)
    model = DataParallel(model)
    metric_fc = metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    # Initialize optimizer
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': metric_fc.parameters()}],
                               lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
