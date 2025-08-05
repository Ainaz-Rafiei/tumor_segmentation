
import os
import cv2
from example import predict

input_dir = 'data/patients/imgs/'
output_dir = 'results/'

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, fname))
    result = predict(img)
    cv2.imwrite(os.path.join(output_dir, fname), result)
