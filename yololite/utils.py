"""
utils.py — Helper functions for YOLO bounding boxes, IoU, and NMS
=================================================================
"""
import torch
import numpy as np

def calculate_iou(boxes1, boxes2):
    """
    Calculates Intersection over Union (IoU) between two sets of bounding boxes.
    Boxes are expected in [x_center, y_center, width, height] format.
    """
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_x2 = boxes1[..., 0] - boxes1[..., 2] / 2, boxes1[..., 0] + boxes1[..., 2] / 2
    b1_y1, b1_y2 = boxes1[..., 1] - boxes1[..., 3] / 2, boxes1[..., 1] + boxes1[..., 3] / 2
    
    b2_x1, b2_x2 = boxes2[..., 0] - boxes2[..., 2] / 2, boxes2[..., 0] + boxes2[..., 2] / 2
    b2_y1, b2_y2 = boxes2[..., 1] - boxes2[..., 3] / 2, boxes2[..., 1] + boxes2[..., 3] / 2

    # Intersection coordinates
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-6

    return inter_area / union_area

def non_max_suppression(predictions, iou_threshold=0.45, conf_threshold=0.25):
    """
    Performs Non-Max Suppression (NMS) on inference results.
    predictions shape: (batch_size, num_boxes, 5 + num_classes)
    Format: [x_center, y_center, width, height, objectness, class_probs...]
    """
    batch_size = predictions.shape[0]
    output = [torch.empty((0, 6), device=predictions.device)] * batch_size

    for i, image_pred in enumerate(predictions):
        # Filter out low confidence boxes
        obj_conf = image_pred[:, 4]
        image_pred = image_pred[obj_conf > conf_threshold]

        if not image_pred.shape[0]:
            continue

        # Get highest class probability
        class_conf, class_pred = torch.max(image_pred[:, 5:], dim=1)
        
        # Filter overall confidence = obj_conf * class_conf
        conf = obj_conf * class_conf
        
        # Convert bounding boxes to [x1, y1, x2, y2]
        boxes = torch.zeros_like(image_pred[:, :4])
        boxes[:, 0] = image_pred[:, 0] - image_pred[:, 2] / 2
        boxes[:, 1] = image_pred[:, 1] - image_pred[:, 3] / 2
        boxes[:, 2] = image_pred[:, 0] + image_pred[:, 2] / 2
        boxes[:, 3] = image_pred[:, 1] + image_pred[:, 3] / 2

        # Final detections format: [x1, y1, x2, y2, conf, class_pred]
        detections = torch.cat((boxes, conf.unsqueeze(1), class_pred.float().unsqueeze(1)), dim=1)

        # Get unique classes
        unique_classes = detections[:, 5].unique()

        for c in unique_classes:
            # Get detections for this class
            class_mask = detections[:, 5] == c
            c_detections = detections[class_mask]

            # Sort by confidence
            sort_indices = torch.argsort(c_detections[:, 4], descending=True)
            c_detections = c_detections[sort_indices]

            # Perform NMS
            keep = []
            while c_detections.shape[0] > 0:
                keep.append(c_detections[0])
                if c_detections.shape[0] == 1:
                    break
                
                # Calculate IoU with rest
                ious = calculate_iou(
                    c_detections[0, :4].unsqueeze(0).repeat(c_detections.shape[0]-1, 1),
                    c_detections[1:, :4]
                )
                
                # Keep boxes with IoU < threshold
                c_detections = c_detections[1:][ious < iou_threshold]

            if keep:
                keep = torch.stack(keep)
                output[i] = torch.cat((output[i], keep))

    return output

def decode_predictions(predictions, anchors, image_size, S=7):
    """
    Decodes network output into actual bounding boxes.
    predictions: (batch_size, num_anchors * (5 + num_classes), S, S)
    """
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    num_classes = predictions.shape[1] // num_anchors - 5
    
    # Reshape: (batch, num_anchors, 5+classes, S, S) -> (batch, S, S, num_anchors, 5+classes)
    predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, S, S).permute(0, 3, 4, 1, 2).contiguous()
    
    # Sigmoid for x, y, objectness, class probabilities
    tx = torch.sigmoid(predictions[..., 0])
    ty = torch.sigmoid(predictions[..., 1])
    to = torch.sigmoid(predictions[..., 4])
    tc = torch.sigmoid(predictions[..., 5:]) # using sigmoid for independent class probabilities
    
    # Exp for width, height
    tw = torch.exp(predictions[..., 2])
    th = torch.exp(predictions[..., 3])
    
    # Create grid x, y
    grid_y, grid_x = torch.meshgrid(torch.arange(S, device=predictions.device), 
                                    torch.arange(S, device=predictions.device), indexing='ij')
    grid_x = grid_x.unsqueeze(2).expand_as(tx)
    grid_y = grid_y.unsqueeze(2).expand_as(ty)
    
    # Prepare anchors
    anchors = torch.tensor(anchors, device=predictions.device).view(1, 1, 1, num_anchors, 2)
    anchor_w = anchors[..., 0]
    anchor_h = anchors[..., 1]
    
    # Calculate bounding box coordinates
    bx = (tx + grid_x) / S
    by = (ty + grid_y) / S
    bw = (tw * anchor_w) / S
    bh = (th * anchor_h) / S
    
    # Stack together
    decoded = torch.zeros_like(predictions)
    decoded[..., 0] = bx
    decoded[..., 1] = by
    decoded[..., 2] = bw
    decoded[..., 3] = bh
    decoded[..., 4] = to
    decoded[..., 5:] = tc
    
    return decoded.view(batch_size, -1, 5 + num_classes)
