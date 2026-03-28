import inference, glob, numpy as np
inference.load_models()
f = open(glob.glob('data/idr_5000/*.jpg')[0], 'rb').read()

# I will recreate the exact logic using the inference variables
img = inference.preprocess_image(f)
print("Img sum:", img.sum())

descriptors, color_hist = inference.get_orb_and_color_features(img)
print("Descriptors shape:", descriptors.shape)

bovw_raw = inference.get_bovw_histogram(descriptors, inference._kmeans_model)
bovw_tfidf = inference._tfidf.transform([bovw_raw]).toarray()[0]
fused = np.hstack((bovw_tfidf, color_hist))
final_features = inference._scaler.transform([fused])
dec = inference._svm.decision_function(final_features)
print("Dec shape:", dec.shape)
print("Dec:", dec)

score = inference.predict_currency(f)
print("Actual result:", score)
