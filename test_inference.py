from ultralytics import YOLO
import cv2

def run_inference():
    # Load your newly trained custom weights
    model = YOLO("runs/detect/pcb_defect_model/weights/best.pt")

    # Run inference on a sample image from your validation set
    source_image = "yolo_dataset/images/val/00041000.jpg" # Update with any valid image path
    
    results = model(source_image, conf=0.25)
    
    # Print detected defects
    for r in results:
        boxes = r.boxes
        print(f"Found {len(boxes)} defects!")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f" - Defect: {class_name} (Confidence: {conf:.2f})")

    # Save or display result image with bounding boxes
    results[0].save(filename="inspection_result.jpg")
    print("Saved visualization to inspection_result.jpg")

if __name__ == "__main__":
    run_inference()