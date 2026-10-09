from ultralytics import YOLO

def train_pcb_model():
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    print("Starting YOLOv8 training on DeepPCB dataset...")
    
    # Train model
    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="pcb_defect_model"
    )
    
    print("\nTraining complete!")
    print("Your model weights are saved at: runs/detect/pcb_defect_model/weights/best.pt")

if __name__ == "__main__":
    train_pcb_model()