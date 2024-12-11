from __future__ import print_function
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from train_model_3 import (
    CustomResNet,
    AudioDataset,
    read_val,
    calculate_mrr
)

def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def train_model_1(opt):
    torch.manual_seed(3407)
    device = torch.device("cuda")

    train_dataset = AudioDataset(opt.train_root, opt.train_list, opt.input_shape)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)}")

    criterion = torch.nn.CrossEntropyLoss()
    model = CustomResNet(feature_dim=512)
    model.to(device)
    model = DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    start = time.time()
    mrr_best = 0
    patience_counter = 0
    max_patience = 5

    for epoch in range(1, opt.max_epoch + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (data_input, label) in enumerate(trainloader):
            try:
                data_input = data_input.to(device)
                label = label.to(device).long()

                feature = model(data_input)
                loss = criterion(feature, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                if batch_idx % opt.print_freq == 0:
                    speed = opt.print_freq / (time.time() - start)
                    print(f'Epoch {epoch} Batch {batch_idx}/{len(trainloader)} '
                          f'Speed {speed:.1f} samples/sec Loss {loss.item():.4f}')
                    start = time.time()

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        scheduler.step()
        epoch_loss /= batch_count
        print(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}")

        try:
            print('Calculating MRR...')
            save_model(model, opt.checkpoints_path, opt.backbone, 'latest')
            model.eval()

            data_val = read_val(opt.val_list, opt.train_root)
            mrr = calculate_mrr(model, data_val, opt.input_shape)
            print(f'Epoch {epoch}: MRR = {mrr:.4f}')

            if mrr > mrr_best:
                mrr_best = mrr
                save_model(model, opt.checkpoints_path, opt.backbone, 'best')
                print(f"New best model saved with MRR: {mrr_best:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print("Early stopping triggered")
                break

        except Exception as e:
            print(f"Error during validation: {e}")
            continue

    print(f"Training completed. Best MRR: {mrr_best:.4f}")
