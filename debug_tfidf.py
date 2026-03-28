import numpy as np
dec = np.array([[-0.24, 4.06, 1.79, 5.23, 0.78, 2.96, 6.26]])
scaled_dec = dec / 0.82
scores = np.exp(scaled_dec) / np.sum(np.exp(scaled_dec), axis=1, keepdims=True)
print("Temp 0.82:", np.max(scores))

scaled_dec = dec / 0.35
scores = np.exp(scaled_dec) / np.sum(np.exp(scaled_dec), axis=1, keepdims=True)
print("Temp 0.35:", np.max(scores))

dec_1000 = np.array([[6.28, 2.88, 0.79, -0.24, 5.10, 1.81, 3.92]])
scaled_dec = dec_1000 / 0.82
scores = np.exp(scaled_dec) / np.sum(np.exp(scaled_dec), axis=1, keepdims=True)
print("Temp 0.82 (1000):", np.max(scores))
