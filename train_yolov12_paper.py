from ultralytics import YOLO
import os

# Set PYTHONPATH to include the modified ultralytics folder if needed
# os.environ["PYTHONPATH"] = os.getcwd()

def train():
    # Load the YOLOv12 model configuration
    model = YOLO("ultralytics/cfg/models/v12/yolov12.yaml") # This will use 's' scale by default if model=yolov12s.yaml or similar
    # But wait, looking at yolov12.yaml, it has 'scales'.
    # To specify scale 's', we can use the 's' scale directly if the API supports it.
    # In modified ultralytics, we usually specify the .yaml and then the scale if it's integrated.
    # If I want YOLOv12-S, I can use:
    model = YOLO("ultralytics/cfg/models/v12/yolov12.yaml", task="detect")
    
    # Define hyperparameters based on YOLOv12 paper
    args = {
        "data": "ultralytics/cfg/datasets/coco.yaml",
        "epochs": 600,
        "batch": 96, 
        "imgsz": 640,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01, # Final LR = lr0 * lrf = 0.0001
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "project": "yolov12_paper_training",
        "name": "yolov12n_coco",
        "device": 0, # Use first GPU
        "workers": 8,
        "exist_ok": True,
        "save_period": 10,
        "close_mosaic": 20, # Default for Ultralytics 600-epoch runs
    }
    
    # YOLOv12-S scale: n, s, m, l, x
    # We should ensure we are using the 's' scale.
    # In some versions, you can pass model="yolov12s.yaml"
    # I'll check if yolov12s.yaml exists or if I should modify the model.yaml
    
    print("Starting training with hyperparameters:", args)
    model.train(**args)

if __name__ == "__main__":
    train()
