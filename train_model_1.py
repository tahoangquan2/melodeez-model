from __future__ import print_function
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from train_model_3 import (
    FocalLoss,
    ResNetFace,
    IRBlock,
    ArcMarginProduct,
    Visualizer,
    AudioDataset,
    read_val,
    mrr_score
)


def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def train_model_1(opt):
    # Set random seed for reproducibility
    torch.manual_seed(3407)

    # Initialize visualization if enabled
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    # Initialize dataset and dataloader
    train_dataset = AudioDataset(opt.train_root, opt.train_list, phase='train',
                                input_shape=opt.input_shape,
                                mp3aug_ratio=opt.mp3aug_ratio, npy_aug=opt.npy_aug)

    trainloader = DataLoader(train_dataset,
                           batch_size=opt.train_batch_size,
                           shuffle=True,
                           num_workers=opt.num_workers)

    # Print dataset info
    print(f"Train dataset size: {len(train_dataset)}")
    print('{} train iters per epoch:'.format(len(trainloader)))

    # Get a sample batch for debugging
    try:
        sample_data = next(iter(trainloader))
        print(f"Batch shape: {sample_data[0].shape}")
        print(f"Label shape: {sample_data[1].shape}")
        print(f"Unique labels: {torch.unique(sample_data[1])}")
    except Exception as e:
        print(f"Warning: Could not get sample batch: {e}")

    # Initialize loss
    criterion = FocalLoss(gamma=2) if opt.loss == 'focal_loss' else torch.nn.CrossEntropyLoss()

    # Initialize model
    print("Initializing model...")
    if opt.backbone == 'resnet18':
        model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=opt.use_se)

    # Initialize metric
    print("Initializing metric...")
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)

    # Move to GPU
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    # Initialize optimizer
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                  lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                   lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    # Training loop
    print("Starting training...")
    start = time.time()
    mrr_best = 0
    global_step = 0

    for epoch in range(1, opt.max_epoch + 1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0

        for batch_idx, (data_input, label) in enumerate(trainloader):
            data_input = data_input.to(device)
            label = label.to(device).long()

            # Forward pass
            try:
                feature = model(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(label.view_as(pred)).float().mean()

                # Update metrics
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                batch_count += 1
                global_step += 1

                # Print progress
                if global_step % opt.print_freq == 0:
                    speed = opt.print_freq / (time.time() - start)
                    time_str = time.asctime(time.localtime(time.time()))
                    print(f'{time_str} Epoch {epoch} Batch {batch_idx}/{len(trainloader)} '
                          f'Speed {speed:.1f} samples/sec Loss {loss.item():.4f} Acc {acc.item():.4f}')

                    if opt.display:
                        visualizer.display_current_results(global_step, loss.item(), name='train_loss')
                    start = time.time()

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        # End of epoch
        scheduler.step()

        # Calculate epoch metrics
        epoch_loss /= batch_count
        epoch_acc /= batch_count
        print(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_acc:.4f}")

        # Validation
        if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
            print('Calculating MRR...')
            save_model(model, opt.checkpoints_path, opt.backbone, 'latest')
            model.eval()

            try:
                data_val = read_val(opt.val_list, opt.train_root)
                mrr = mrr_score(model, data_val, opt.input_shape)
                print(f'Epoch {epoch}: MRR = {mrr:.4f}')

                if mrr > mrr_best:
                    mrr_best = mrr
                    save_model(model, opt.checkpoints_path, opt.backbone, 'best')
                    print(f"New best model saved with MRR: {mrr_best:.4f}")
            except Exception as e:
                print(f"Error during validation: {e}")
                continue

    print(f"Training completed. Best MRR: {mrr_best:.4f}")
