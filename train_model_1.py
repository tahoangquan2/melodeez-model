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
                               drop_last=True)

        val_dataset = AudioDataset(opt.train_root, opt.val_list, opt.input_shape)
        valloader = DataLoader(val_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")
        logger.info(f"{len(trainloader)} train iters per epoch")

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

        train_loss = epoch_loss / batch_count

        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for data_input, label in valloader:
                try:
                    data_input = data_input.to(device)
                    label = label.to(device).long()
                    feature = model(data_input)
                    loss = criterion(feature, label)
                    val_loss += loss.item()
                    val_count += 1
                except Exception as e:
                    continue

        val_loss = val_loss / val_count if val_count > 0 else float('inf')

        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, opt.checkpoints_path, opt.backbone, 'best')
            logger.info(f"New best model saved with val loss: {best_loss:.4f}")

        save_model(model, opt.checkpoints_path, opt.backbone, 'latest')
        scheduler.step()
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    logger.info(f"Training completed. Best val loss: {best_loss:.4f}")
    return best_loss
