from ultralytics import YOLO


# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
# from ultralytics import settings
# print(settings)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
# results = model.val(data='coco128.yaml')

# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# # Export the model to ONNX format
# success = model.export(format='onnx')

# from ultralytics.utils.benchmarks import benchmark

# # Benchmark
# benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)