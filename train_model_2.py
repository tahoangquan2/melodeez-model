from __future__ import print_function
import os
import glob

def generate_lists(root_dir='output/output3', train_ratio=0.8, output_dir='checkpoints'):
    os.makedirs(output_dir, exist_ok=True)

    hum_files = glob.glob(os.path.join(root_dir, 'hum', '*.npy'))
    song_files = glob.glob(os.path.join(root_dir, 'song', '*.npy'))

    hum_ids = sorted(list(set([int(os.path.basename(f).split('.')[0].split('_')[0]) for f in hum_files])))

    num_train = int(len(hum_ids) * train_ratio)
    train_ids = set(hum_ids[:num_train])
    val_ids = set(hum_ids[num_train:])

    train_lines = []
    val_lines = []

    for f in hum_files:
        filename = os.path.basename(f)
        file_id = int(filename.split('.')[0].split('_')[0])
        rel_path = os.path.join('hum', filename)
        line = f"{rel_path} {file_id}"
        if file_id in train_ids:
            train_lines.append(line)
        else:
            val_lines.append(line)

    for f in song_files:
        filename = os.path.basename(f)
        file_id = int(filename.split('.')[0].split('_')[0])
        rel_path = os.path.join('song', filename)
        line = f"{rel_path} {file_id}"
        if file_id in train_ids:
            train_lines.append(line)
        else:
            val_lines.append(line)

    with open(os.path.join(output_dir, 'train_list.txt'), 'w') as f:
        f.write('\n'.join(train_lines))

    with open(os.path.join(output_dir, 'val_list.txt'), 'w') as f:
        f.write('\n'.join(val_lines))

class Config:
    def __init__(self):
        # Training settings
        self.train_batch_size = 64 # 32
        self.num_workers = 4
        self.max_epoch = 100
        self.lr = 1e-4 # 3e-4
        self.weight_decay = 0.01 # 0.0005
        self.print_freq = 100

        # Model settings
        self.backbone = 'resnet18'
        self.input_shape = (1, 80, 630)
        self.num_classes = 20 # Unique songs
        self.max_patience = 10 # 5
        self.val_freq = 3 # 5

        # Data settings
        self.train_root = 'output/output3'
        self.train_list = 'checkpoints/train_list.txt'
        self.val_list = 'checkpoints/val_list.txt'

        # Save settings
        self.checkpoints_path = 'checkpoints'

        # DO NOT CHANGE I DONT KNOW WHY BUT DONT - Visualization settings
        self.display = False

def main():
    opt = Config()
    os.makedirs(opt.checkpoints_path, exist_ok=True)

    if not os.path.exists(opt.train_list) or not os.path.exists(opt.val_list):
        print("Generating train and validation lists...")
        generate_lists(opt.train_root)

    from train_model_1 import train_model_1
    train_model_1(opt)

if __name__ == '__main__':
    main()
