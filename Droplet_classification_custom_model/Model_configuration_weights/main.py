import cv2
import numpy as np
import os
import time

# Load Network
net = cv2.dnn.readNet("dnn_model/yolov4-tiny-custom_6000.weights", "dnn_model/yolov4-tiny-custom.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416 ), scale=1/255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Path to the folder containing images
images_folder = "Testing/"

# Output folder to save processed images
output_folder = "Processed_Images"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List to store paths of processed images
processed_image_paths = []

# Initialize variables for batch processing
batch_size = 6
current_batch = []
batch_index = 0

# Iterate over all images in the folder
for image_name in os.listdir(images_folder):
    # Load Image
    img = cv2.imread(os.path.join(images_folder, image_name))
    current_batch.append(img)

 # Check if the current batch is full or it's the last image
    if len(current_batch) == batch_size or image_name == os.listdir(images_folder)[-1]:

        # Perform object detection on the batch
        for img_index, img in enumerate(current_batch):
            start_time = time.time()
            class_ids, scores, boxes = model.detect(img, nmsThreshold=0.7)

            for (class_id, score, box) in zip(class_ids, scores, boxes):
                x, y, w, h = box
                class_name = classes[class_id[0]]
                color = colors[class_id[0]]
                cv2.putText(img, "{} {}".format(class_name, score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 1)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

            # Save the processed image
            output_path = os.path.join(output_folder, f"processed_image_{batch_index * batch_size + img_index}.jpg")
            cv2.imwrite(output_path, img)
            processed_image_paths.append(output_path)

        # End time for the batch
        # end_time = time.time()
        # Compute processing time for the batch
        # processing_time = end_time - start_time
        #         print("Processing time for each image in batch {}: {:.2f} seconds".format(batch_index, processing_time))

        # Clear the current batch for the next iteration
        current_batch = []
        batch_index += 1
        end_time = time.time()
        # Compute processing time for the batch
        processing_time = end_time - start_time
        print("Processing time for each image in batch {}: {:.2f} seconds".format(batch_index, processing_time))
# Close all OpenCV windows
cv2.destroyAllWindows()

# Display all processed images
for path in processed_image_paths:
    img = cv2.imread(path)
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()