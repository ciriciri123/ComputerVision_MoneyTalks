import inference, glob, numpy as np, cv2
inference.load_models()
ood_v = np.array([6.2098, 5.2043, 1.9318, -0.2344, 3.0532, 0.8095, 4.1091])
img = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
features = inference._scaler.transform([np.hstack((inference._tfidf.transform([inference.get_bovw_histogram(inference.get_orb_and_color_features(img)[0], inference._kmeans_model)]).toarray()[0], inference.get_orb_and_color_features(img)[1]))])
dec = inference._svm.decision_function(features)[0]
print("Noise dist:", np.linalg.norm(dec - ood_v))
