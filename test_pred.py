import inference, glob, numpy as np, cv2

# Real image
f = open(glob.glob('data/idr_50000/*.jpg')[0], 'rb').read()
print("Real:", inference.predict_currency(f))

# Blank image
blank = np.zeros((400, 800, 3), dtype=np.uint8)
_, encoded = cv2.imencode('.jpg', blank)
print("Blank:", inference.predict_currency(encoded.tobytes()))

# Random noise
noise = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
_, encoded_n = cv2.imencode('.jpg', noise)
print("Noise:", inference.predict_currency(encoded_n.tobytes()))
