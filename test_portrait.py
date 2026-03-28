import inference, glob, numpy as np, cv2
img_path = glob.glob('data/idr_50000/*.jpg')[0]
img = cv2.imread(img_path)
# Rotate to portrait
img_portrait = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
res = inference.predict_currency(cv2.imencode('.jpg', img_portrait)[1].tobytes())
print("Portrait 50000 ->", res)
