"""
inference.py — Standalone inference and demo script
===================================================

Runs inference on a single image and measures FPS.
"""

import torch
import cv2
import numpy as np
import time
import argparse
from yololite.model import YOLOLite
from yololite.utils import non_max_suppression, decode_predictions
from yololite.dataset import VOC_CLASSES

ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52)
]

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-LITE Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .pt file or Darknet .weights")
    parser.add_argument("--activation", type=str, default="leaky", help="Activation used during training")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = YOLOLite(num_classes=20, activation_name=args.activation)
    
    if args.weights.endswith(".weights"):
        print("Loading Darknet weights...")
        model.load_darknet_weights(args.weights)
    else:
        print("Loading PyTorch checkpoint...")
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
    model.to(device)
    model.eval()

    # Load and preprocess image
    orig_img = cv2.imread(args.image)
    if orig_img is None:
        raise ValueError(f"Could not read image: {args.image}")
        
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    print("Running inference...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(img_tensor)

    # Actual inference with timing
    start_time = time.time()
    with torch.no_grad():
        pred_raw = model(img_tensor)
        
        # Decode and NMS
        pred_decoded = decode_predictions(pred_raw, ANCHORS, 224, S=7)
        detections = non_max_suppression(pred_decoded, iou_threshold=args.iou, conf_threshold=args.conf)
        
    end_time = time.time()
    print(f"Inference Time (including NMS): {(end_time - start_time)*1000:.1f} ms")
    print(f"FPS: {1.0 / (end_time - start_time):.1f}")

    # Draw bounding boxes
    h, w, _ = orig_img.shape
    if detections[0] is not None and len(detections[0]) > 0:
        for x1, y1, x2, y2, conf, cls_pred in detections[0]:
            # Scale back to original image size
            x1 = int(x1.item() * w)
            y1 = int(y1.item() * h)
            x2 = int(x2.item() * w)
            y2 = int(y2.item() * h)
            cls_id = int(cls_pred.item())
            
            label = f"{VOC_CLASSES[cls_id]}: {conf.item():.2f}"
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detected: {label} at [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("No objects detected.")

    # Save output
    out_path = "output_" + args.image.split("/")[-1]
    cv2.imwrite(out_path, orig_img)
    print(f"Saved result to {out_path}")

if __name__ == "__main__":
    main()
