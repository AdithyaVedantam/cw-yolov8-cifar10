from ultralytics import YOLO

model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

metrics = model.val(
    data="cifar10"
)

print(f"\nClean Accuracy: {metrics.top1:.2%}")