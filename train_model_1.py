from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
import time
from torch.nn import DataParallel
from train_model_3 import (
    ResNetFace,
    AudioDataset,
    FocalLoss,
)
from logger import logger

def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def train_model_1(opt):
    torch.manual_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        train_dataset = AudioDataset(opt.train_root, opt.train_list, opt.input_shape)
        trainloader = DataLoader(train_dataset,
                               batch_size=opt.train_batch_size,
                               shuffle=True,
                               num_workers=opt.num_workers,
                               pin_memory=True,
                               persistent_workers=True,
                               prefetch_factor=2)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"{len(trainloader)} train iters per epoch")

        sample_data = next(iter(trainloader))
        logger.info(f"Batch shape: {sample_data[0].shape}")
        logger.info(f"Label shape: {sample_data[1].shape}")
        unique_labels = torch.unique(sample_data[1])
        logger.info(f"Unique labels: {unique_labels}")
        logger.info(f"Label range: {unique_labels.min().item()} to {unique_labels.max().item()}")

    except Exception as e:
        logger.error(f"Fatal error in data loading: {e}")
        return None

    logger.info("Initializing model...")
    model = ResNetFace(feature_dim=opt.embedding_dim)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=0.9,
        weight_decay=opt.weight_decay
    )

    criterion = FocalLoss(gamma=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.max_epoch,
        eta_min=opt.lr * 1e-4
    )

    logger.info("Starting training...")
    start = time.time()
    best_loss = float('inf')

    failed_batches = 0
    max_failed_batches = len(trainloader) // 2

    for epoch in range(1, opt.max_epoch + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        failed_batches = 0

        for batch_idx, (data_input, label) in enumerate(trainloader):
            try:
                if torch.isnan(data_input).any() or torch.isinf(data_input).any():
                    raise ValueError("Input contains NaN or Inf values")
                if label.min() < 0:
                    raise ValueError(f"Negative label found: {label.min()}")

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
                    logger.info(f'{time_str} Epoch {epoch} Batch {batch_idx}/{len(trainloader)} '
                              f'Speed {speed:.1f} samples/sec Loss {loss.item():.4f}')
                    start = time.time()

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                failed_batches += 1
                if failed_batches > max_failed_batches:
                    logger.error("Too many failed batches, stopping training")
                    return None
                continue

        if batch_count == 0:
            logger.error("No successful batches in epoch, stopping training")
            return None

        epoch_loss /= batch_count
        logger.info(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}, Failed Batches: {failed_batches}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model(model, opt.checkpoints_path, opt.backbone, 'best')
            logger.info(f"New best model saved with loss: {best_loss:.4f}")

        save_model(model, opt.checkpoints_path, opt.backbone, 'latest')
        scheduler.step()
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    logger.info(f"Training completed. Best loss: {best_loss:.4f}")
    return best_loss
