from groundingdino.util.inference import Model
from constants import *
import torchvision
import torch
import numpy as np
def initialize_dino_model(dino_config=GROUNDING_DINO_CONFIG_PATH,
                          dino_checkpoint=GROUNDING_DINO_CHECKPOINT_PATH,
                          device="cuda:0"):
    model = Model(model_config_path=dino_config,
                  model_checkpoint_path=dino_checkpoint,
                  device=device)
    return model

def openset_detection(image,target_classes,dino_model,
                      box_threshold=0.2,
                      text_threshold=0.4,
                      nms_threshold=0.5):
    detections = dino_model.predict_with_classes(image=image,
                                                classes=target_classes,
                                                box_threshold=box_threshold,
                                                text_threshold=text_threshold)
    detections.xyxy = detections.xyxy[detections.class_id!=None]
    detections.confidence = detections.confidence[detections.class_id!=None]
    detections.class_id = detections.class_id[detections.class_id!=None]
    nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy), 
                                torch.from_numpy(detections.confidence), 
                                nms_threshold).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    return detections
def openset_detection_yolov11(image, target_classes, yolov11_model, 
                              conf_threshold=0.25, iou_threshold=0.45):
    """
    Perform open-set detection with YOLOv11, filtering for specific target classes and applying NMS.

    Args:
        image (str or np.ndarray): Path to the image or a loaded image.
        target_classes (list): List of target class names to detect.
        yolov11_model (torch.nn.Module): Loaded YOLOv11 model.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IOU threshold for NMS.

    Returns:
        dict: Filtered detections with bounding boxes, class IDs, and confidence scores.
    """
    # Run inference
    results = yolov11_model(image)

    # Get class names from the model
    class_names = yolov11_model.names

    # Map target class names to their corresponding IDs
    target_class_ids = [i for i, name in enumerate(class_names) if name in target_classes]

    # Iterate through each result in the output
    filtered_detections = []
    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = res.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = res.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Filter detections by confidence threshold and target classes
        keep = (confidences >= conf_threshold) & np.isin(class_ids, target_class_ids)
        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # Apply Non-Maximum Suppression (NMS)
        keep_nms = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(confidences), iou_threshold).numpy()
        boxes = boxes[keep_nms]
        confidences = confidences[keep_nms]
        class_ids = class_ids[keep_nms]

        # Store filtered detections
        filtered_detections.append({
            "boxes": boxes,  # Bounding boxes
            "confidences": confidences,  # Confidence scores
            "class_ids": class_ids,  # Class IDs
            "class_names": [class_names[i] for i in class_ids]  # Class names
        })

    return filtered_detections
