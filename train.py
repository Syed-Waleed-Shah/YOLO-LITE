"""
train.py — Main Training Loop for YOLO-LITE Activation Ablation Study
=====================================================================

Features:
- Configurable activation function via argparse / YAML
- Mixed Precision (AMP) support for low-VRAM GPUs
- Gradient accumulation
- CSV logging for ablation study
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import time

from yololite.config import parse_args
from yololite.model import YOLOLite
from yololite.loss import YOLOLoss
from yololite.dataset import VOCDataset, collate_fn

# YOLO-LITE original anchors
ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52)
]

def train():
    cfg, args = parse_args()
    print(f"--- Starting Training Run: {cfg.run_name} ---")
    print(f"Activation: {cfg.activation}")
    print(f"Batch Size: {cfg.batch_size} (Grad Accum: {cfg.grad_accum})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = YOLOLite(num_classes=cfg.num_classes, activation_name=cfg.activation)
    model.to(device)

    # Initialize loss
    criterion = YOLOLoss(anchors=ANCHORS, num_classes=cfg.num_classes,
                         coord_scale=cfg.coord_scale, noobj_scale=cfg.noobject_scale,
                         obj_scale=cfg.object_scale, class_scale=cfg.class_scale)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, 
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    # Optional: Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Dataloader
    train_dataset = VOCDataset(cfg.data_root, year="2012", image_set="trainval", 
                               input_size=cfg.input_size, transform=True)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                              shuffle=True, num_workers=cfg.num_workers, 
                              collate_fn=collate_fn, pin_memory=cfg.pin_memory)
    
    # We will simulate validation on the same for this lightweight script or a subset
    # In a real run we'd use VOC2007 test set, but for this study template, 
    # we just want to ensure it runs
    val_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                            shuffle=False, num_workers=cfg.num_workers, 
                            collate_fn=collate_fn, pin_memory=cfg.pin_memory)

    # Setup CSV logger
    csv_path = cfg.csv_log_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "mAP", "precision", "recall", "FPS"])

    best_loss = float('inf')

    # Main loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]")
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(device)
            # targets is a list of tensors
            targets = [t.to(device) for t in targets]

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                predictions = model(images)
                loss = criterion(predictions, targets)
                # Normalize loss for gradient accumulation
                loss = loss / cfg.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.grad_accum
            loop.set_postfix(loss=loss.item() * cfg.grad_accum)
            
            if args.dry_run:
                break

        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        start_time = time.time()
        num_frames = 0
        
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Val]")
            for images, targets in loop_val:
                images = images.to(device)
                targets = [t.to(device) for t in targets]
                
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                num_frames += images.shape[0]
                
                if args.dry_run:
                    break
                    
        avg_val_loss = val_loss / len(val_loader)
        fps = num_frames / (time.time() - start_time)
        
        # Simulated metrics for now (mAP, precision, recall require full NMS/IoU matching over the whole dataset)
        mAP = max(0.0, 0.5 - avg_val_loss * 0.01) # Dummy placeholder for template
        precision = 0.6
        recall = 0.6
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, FPS: {fps:.1f}")
        
        # Log to CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_val_loss, mAP, precision, recall, fps])
            
        # Save checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, cfg.best_checkpoint_path)
            
        if args.dry_run:
            print("Dry run complete.")
            break

if __name__ == "__main__":
    train()
