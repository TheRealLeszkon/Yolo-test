from ultralytics import YOLO

model = YOLO("models/objects365/yoloe-11s-seg-pf.pt")

model.export(
    format="onnx",
    opset=17,
    simplify=True,
    dynamic=False
)