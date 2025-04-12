from ultralytics import YOLO

# Load the YOLOv8 model (pretrained weights)
model = YOLO('yolov8m.pt')  # 'n' for nano; use 's', 'm', 'l', or 'x' for larger models

# Train the model
model.train(
    data='/home/msi/course/atherenergydataset/dataset/data.yaml',  # Path to your data.yaml
    epochs=25,                 # Number of epochs
    imgsz=640,                 # Image size (adjust if needed)
    batch=16,                  # Batch size (adjust based on GPU memory)
    name='ather_scooter_run',  # Name of the training run
    patience=10,               # Early stopping patience
    device= 'cpu'                   # Use GPU (0) or 'cpu' if no GPU
)

# Save the trained model
model.save('ather_scooter_model.pt')
