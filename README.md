import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load custom dataset (replace with your dataset loading code)
# Assuming the dataset contains images and annotations
# You can use OpenCV or any other library to load images

# Example:
# images = []
# annotations = []

# for image_path, annotation_path in zip(image_paths, annotation_paths):
#     image = cv2.imread(image_path)
#     images.append(image)
#     annotations.append(annotation_path)

# Function to perform object detection on an image
def detect_objects(image):
    # Resize image and normalize pixel values
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pass input image through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Example usage:
# for image_path in image_paths:
#     image = cv2.imread(image_path)
#     detected_image = detect_objects(image)
#     cv2.imshow('Object Detection', detected_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
