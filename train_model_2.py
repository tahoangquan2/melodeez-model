from __future__ import print_function
import os
import torch
import glob

def generate_lists(root_dir='output/output3', train_ratio=0.8):
    hum_files = glob.glob(os.path.join(root_dir, 'hum', '*.npy'))
    song_files = glob.glob(os.path.join(root_dir, 'song', '*.npy'))

    # Extract unique IDs
    hum_ids = sorted(list(set([int(os.path.basename(f).split('.')[0].split('_')[0]) for f in hum_files])))

    num_train = int(len(hum_ids) * train_ratio)
    train_ids = set(hum_ids[:num_train])
    val_ids = set(hum_ids[num_train:])

    train_lines = []
    val_lines = []

    # Process hum files
    for f in hum_files:
        filename = os.path.basename(f)
        file_id = int(filename.split('.')[0].split('_')[0])
        rel_path = os.path.join('hum', filename)
        line = f"{rel_path} {file_id}"
        if file_id in train_ids:
            train_lines.append(line)
        else:
            val_lines.append(line)


    # Process song files
    for f in song_files:
        filename = os.path.basename(f )
        file_id = int(filename.split('.')[0].split('_')[0])
        rel_path = os.path.join('song', filename)
        line = f"{rel_path} {file_id}"
        if file_id in train_ids:
            train_lines.append(line)
        else:
            val_lines.append(line)

    # Write files
    with open('train_list.txt', 'w') as f:
        f.write('\n'.join(train_lines))

    with open('val_list.txt', 'w') as f:
        f.write('\n'.join(val_lines))

class Config:
    def __init__(self):
        # Training settings
        self.train_batch_size = 32
        self.num_workers = 4
        self.max_epoch = 20
        self.lr = 0.1
        self.weight_decay = 0.0005
        self.lr_step = 10
        self.print_freq = 100
        self.save_interval = 1

        # Model settings
        self.backbone = 'resnet18'
        self.use_se = True
        self.metric = 'arc_margin'
        self.easy_margin = False
        self.loss = 'focal_loss'
        self.optimizer = 'sgd'
        # Number of unique songs
        self.num_classes = 12

        # Data settings
        self.input_shape = (1, 80, 630)
        self.train_root = 'output/output3'
        self.train_list = 'train_list.txt'
        self.val_list = 'val_list.txt'
        self.mp3aug_ratio = 0.5
        self.npy_aug = True

        # Save settings
        self.checkpoints_path = 'checkpoints'
        # Disable visualization
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
