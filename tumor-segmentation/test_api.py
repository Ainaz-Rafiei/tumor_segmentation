import requests
import cv2
from utils import encode_request, decode_request

img = cv2.imread("data/patients/imgs/patient_001.png")
encoded_img = encode_request(img)

response = requests.post("http://localhost:9051/predict", json={"img": encoded_img})

assert response.status_code == 200
result = response.json()
assert "img" in result

decoded_mask = decode_request({"img": result["img"]})
assert decoded_mask.shape == img.shape
