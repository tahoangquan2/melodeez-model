from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
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

    # Implement k-fold cross validation
    k_folds = 5
    train_dataset = AudioDataset(opt.train_root, opt.train_list, opt.input_shape)
    dataset_size = len(train_dataset)
    fold_size = dataset_size // k_folds

    fold_scores = []

    for fold in range(k_folds):
        logger.info(f"Starting fold {fold + 1}/{k_folds}")

        val_start = fold * fold_size
        val_end = val_start + fold_size

        train_indices = list(range(0, val_start)) + list(range(val_end, dataset_size))
        val_indices = list(range(val_start, val_end))

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        trainloader = DataLoader(train_subset,
                               batch_size=opt.train_batch_size,
                               shuffle=True,
                               num_workers=opt.num_workers,
                               pin_memory=True,
                               drop_last=True)

        valloader = DataLoader(val_subset,
                             batch_size=opt.train_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)

        logger.info(f"Train subset size: {len(train_subset)}")
        logger.info(f"Val subset size: {len(val_subset)}")

        model = ResNetFace(feature_dim=opt.embedding_dim)
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.lr * 0.1,
            weight_decay=opt.weight_decay
        )

        criterion = FocalLoss(gamma=2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=3,
            min_lr=opt.lr * 1e-5
        )

        best_loss = float('inf')
        patience = 0
        max_patience = 10

        for epoch in range(1, opt.max_epoch + 1):
            model.train()
            epoch_loss = 0
            batch_count = 0

            for batch_idx, (data_input, label) in enumerate(trainloader):
                try:
                    if torch.isnan(data_input).any() or torch.isinf(data_input).any():
                        continue
                    if label.min() < 0:
                        continue

                    data_input = data_input.to(device)
                    label = label.to(device).long()

                    feature = model(data_input)
                    loss = criterion(feature, label)

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                    if batch_idx % opt.print_freq == 0:
                        logger.info(f'Fold {fold + 1} Epoch {epoch} Batch {batch_idx}/{len(trainloader)} '
                                  f'Loss {loss.item():.4f}')

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue

            train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

            # Validation phase
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

            logger.info(f"Fold {fold + 1} Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Learning rate scheduling based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                save_name = save_model(model, opt.checkpoints_path, f"{opt.backbone}_fold{fold + 1}", 'best')
                logger.info(f"New best model saved: {save_name}")
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        fold_scores.append(best_loss)
        logger.info(f"Fold {fold + 1} completed. Best val loss: {best_loss:.4f}")

    mean_score = sum(fold_scores) / len(fold_scores)
    std_score = (sum((x - mean_score) ** 2 for x in fold_scores) / len(fold_scores)) ** 0.5

    logger.info("Cross-validation completed:")
    logger.info(f"Mean validation loss: {mean_score:.4f} ± {std_score:.4f}")
    logger.info(f"Individual fold scores: {[f'{x:.4f}' for x in fold_scores]}")

    return mean_score, std_score
