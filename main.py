import cv2
import numpy as np
import pandas as pd

def detect_dogs_in_video(video_path, weights_path, config_path, output_video_path, output_csv_path):
    # Load names of classes
    classes = None
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Load Yolo
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten().tolist()]

    # Loading video
    cap = cv2.VideoCapture(video_path)

    # Set backend and target to CUDA to use GPU
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Get the properties of video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame to get the size
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    height, width, channels = frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind the video to the beginning

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    frame_count = 0
    detection_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for detection_out in outs:
            for detection in detection_out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == 'dog':
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    detection_list.append(frame_count / frame_rate)

        # write the frame
        out.write(frame)

        frame_count += 1
        print(f"Processed {frame_count} of {total_frames} frames.")  # print progress

    # Release everything after the job is finished
    cap.release()
    out.release()  # don't forget to release the VideoWriter
    cv2.destroyAllWindows()

    # Write the dog detection time to CSV
    df = pd.DataFrame(detection_list, columns=['Time'])
    df.to_csv(output_csv_path, index=False)

# Call the function with output paths
# detect_dogs_in_video('videoplayback.mp4', 'yolov3.weights', 'yolov3.cfg', 'output.mp4', 'dog_detection.csv')
