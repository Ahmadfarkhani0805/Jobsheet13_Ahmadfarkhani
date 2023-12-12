import cv2
import numpy as np

# Membaca model YOLO
model = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Membaca gambar
img = cv2.imread('dogandperson.jpeg')

# Mengubah gambar ke bentuk blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Menjalankan model YOLO
model.setInput(blob)
out = model.forward()

# Mengambil bounding boxes dari hasil deteksi
boxes = out[1,1]
confidences = out[1,1]
classes = out[1,1]

# Mengambil bounding boxes dengan confidence lebih dari 0.5
idxs = np.where(confidences > 0.5)[0]

# Menampilkan bounding boxes
for i in idxs:
    x, y, w, h = boxes[i]
    class_id = classes[i]
    confidence = confidences[i]

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Menampilkan gambar
cv2.imshow('image', img)
cv2.waitKey(0)
