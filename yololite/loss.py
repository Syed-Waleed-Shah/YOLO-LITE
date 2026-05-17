"""
loss.py — YOLO v2 Detection Loss
=================================

Calculates YOLO v2 loss:
1. Coordinate Loss (MSE) - only for responsible anchors
2. Objectness Loss (BCE) - for all anchors (1 if obj exists, 0 otherwise)
3. Classification Loss (BCE / CrossEntropy) - only for responsible anchors
"""

import torch
import torch.nn as nn
from yololite.utils import calculate_iou

class YOLOLoss(nn.Module):
    def __init__(self, anchors, S=7, num_classes=20, 
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=5.0, class_scale=1.0):
        super().__init__()
        self.anchors = anchors # list of tuples: [(w1, h1), (w2, h2), ...]
        self.num_anchors = len(anchors)
        self.S = S
        self.num_classes = num_classes
        
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale

        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_anchors * (5 + num_classes), S, S)
        targets: list of length batch_size, where each element is a tensor of shape (num_objects, 5)
                 format: [class_id, x_center, y_center, width, height] (normalized 0-1)
        """
        batch_size = predictions.shape[0]
        
        # Reshape: (batch, num_anchors, 5+classes, S, S) -> (batch, S, S, num_anchors, 5+classes)
        predictions = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, self.S, self.S)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()
        
        # Target matrices
        obj_mask = torch.zeros((batch_size, self.S, self.S, self.num_anchors), device=predictions.device)
        noobj_mask = torch.ones((batch_size, self.S, self.S, self.num_anchors), device=predictions.device)
        target_coords = torch.zeros((batch_size, self.S, self.S, self.num_anchors, 4), device=predictions.device)
        target_classes = torch.zeros((batch_size, self.S, self.S, self.num_anchors), dtype=torch.long, device=predictions.device)
        
        # Pre-calculate anchor bounding boxes centered at origin for IoU matching
        anchor_boxes = torch.zeros((self.num_anchors, 4), device=predictions.device)
        for i, (w, h) in enumerate(self.anchors):
            # Format: x, y, w, h
            anchor_boxes[i, 2] = w / self.S
            anchor_boxes[i, 3] = h / self.S
        
        for b in range(batch_size):
            for t in targets[b]:
                if t.numel() == 0:
                    continue
                class_id, gx, gy, gw, gh = t
                
                # Grid cell index
                gi = int(gx * self.S)
                gj = int(gy * self.S)
                
                if gi >= self.S or gj >= self.S:
                    continue

                # Target box centered at origin
                target_box = torch.tensor([0, 0, gw, gh], device=predictions.device)
                
                # Find best anchor
                best_iou = 0
                best_anchor = -1
                for i in range(self.num_anchors):
                    iou = calculate_iou(target_box.unsqueeze(0), anchor_boxes[i].unsqueeze(0))
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = i
                
                if best_anchor == -1: continue

                obj_mask[b, gj, gi, best_anchor] = 1
                noobj_mask[b, gj, gi, best_anchor] = 0
                
                # tx, ty should be offsets from cell top-left
                tx = gx * self.S - gi
                ty = gy * self.S - gj
                
                # tw, th should be log(box_size / anchor_size)
                aw, ah = self.anchors[best_anchor]
                tw = torch.log(gw * self.S / aw + 1e-16)
                th = torch.log(gh * self.S / ah + 1e-16)
                
                target_coords[b, gj, gi, best_anchor] = torch.tensor([tx, ty, tw, th], device=predictions.device)
                target_classes[b, gj, gi, best_anchor] = class_id

        # Extracted predicted coordinates
        pred_xy = torch.sigmoid(predictions[..., 0:2])
        pred_wh = predictions[..., 2:4]
        pred_conf = predictions[..., 4]
        pred_classes = predictions[..., 5:]

        # Coordinate loss
        coord_loss = self.mse_loss(obj_mask.unsqueeze(-1) * pred_xy, obj_mask.unsqueeze(-1) * target_coords[..., 0:2]) + \
                     self.mse_loss(obj_mask.unsqueeze(-1) * pred_wh, obj_mask.unsqueeze(-1) * target_coords[..., 2:4])
                     
        # Objectness loss
        obj_loss = self.bce_loss(pred_conf[obj_mask == 1], torch.ones_like(pred_conf[obj_mask == 1]))
        noobj_loss = self.bce_loss(pred_conf[noobj_mask == 1], torch.zeros_like(pred_conf[noobj_mask == 1]))
        conf_loss = self.obj_scale * obj_loss + self.noobj_scale * noobj_loss

        # Classification loss
        if obj_mask.sum() > 0:
            class_loss = self.ce_loss(pred_classes[obj_mask == 1], target_classes[obj_mask == 1])
        else:
            class_loss = torch.tensor(0.0, device=predictions.device)

        total_loss = self.coord_scale * coord_loss + conf_loss + self.class_scale * class_loss
        
        # Average over batch
        return total_loss / batch_size
