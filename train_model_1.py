from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from train_model_3 import (
    CustomResNet,
    AudioDataset,
    read_val,
    calculate_mrr,
    Visualizer
)

def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def train_model_1(opt):
    torch.manual_seed(3407)

    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = AudioDataset(opt.train_root, opt.train_list, opt.input_shape)
    trainloader = DataLoader(train_dataset,
                           batch_size=opt.train_batch_size,
                           shuffle=True,
                           num_workers=opt.num_workers,
                           pin_memory=True,
                           persistent_workers=True,
                           prefetch_factor=2)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"{len(trainloader)} train iters per epoch")

    try:
        sample_data = next(iter(trainloader))
        print(f"Batch shape: {sample_data[0].shape}")
        print(f"Label shape: {sample_data[1].shape}")
        print(f"Unique labels: {torch.unique(sample_data[1])}")
    except Exception as e:
        print(f"Warning: Could not get sample batch: {e}")

    print("Initializing model...")
    model = CustomResNet(feature_dim=opt.num_classes)
    model.to(device)
    model = DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    print("Starting training...")
    start = time.time()
    mrr_best = 0
    patience_counter = 0
    max_patience = opt.max_patience

    for epoch in range(1, opt.max_epoch + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (data_input, label) in enumerate(trainloader):
            # print(f"Unique labels in batch: {torch.unique(label)}")
            # print(f"Min label: {label.min()}, Max label: {label.max()}")

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
                    time_str = time.asctime(time.localtime(time.time()))
                    print(f'{time_str} Epoch {epoch} Batch {batch_idx}/{len(trainloader)} '
                          f'Speed {speed:.1f} samples/sec Loss {loss.item():.4f}')

                    if opt.display:
                        visualizer.display_current_results(batch_idx + epoch*len(trainloader),
                                                        loss.item(),
                                                        name='train_loss')
                    start = time.time()

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        epoch_loss /= batch_count
        print(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}")

        if epoch % opt.val_freq == 0 or epoch == opt.max_epoch:
            try:
                print('Calculating MRR...')
                save_model(model, opt.checkpoints_path, opt.backbone, 'latest')
                model.eval()

                data_val = read_val(opt.val_list, opt.train_root)
                mrr = calculate_mrr(model, data_val, opt.input_shape)
                print(f'Epoch {epoch}: MRR = {mrr:.4f}')
                scheduler.step()

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
    return mrr_best
