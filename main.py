from ultralytics import YOLO
import cv2 as cv


model = YOLO("yolo11n.pt")

results = model("images/Desk.jpeg")

for result in results:
    for box in result.boxes:

        class_id = int(box.cls)
        confidence = float(box.conf)

        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0]

        print(
            class_name,
            confidence,
            (x1.item(), y1.item(), x2.item(), y2.item())
        )

annotated = results[0].plot()
cv.imwrite("output.jpg", annotated)

import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load model
sam = sam_model_registry["vit_h"](
    checkpoint="sam_vit_h_4b8939.pth"
)

# Create automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Load image
image = cv.imread("images/Desk.jpeg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Generate masks
masks = mask_generator.generate(image)

print(f"Found {len(masks)} masks")

import matplotlib.pyplot as plt

plt.imshow(image)

for mask in masks:
    seg = mask["segmentation"]

    overlay = np.zeros(
        (*seg.shape, 4),
        dtype=np.float32
    )

    overlay[seg] = [1, 0, 0, 0.3]

    plt.imshow(overlay)

plt.show()