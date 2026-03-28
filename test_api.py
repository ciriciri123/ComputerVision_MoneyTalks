import requests, glob

img_path = glob.glob('data/idr_50000/*.jpg')[0]
with open(img_path, 'rb') as f:
    resp = requests.post('http://127.0.0.1:5000/api/detect', files={'frame': ('frame.jpg', f, 'image/jpeg')})
    print("API Response:", resp.status_code, resp.text)
