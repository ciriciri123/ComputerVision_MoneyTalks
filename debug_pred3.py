import inference, glob, numpy as np
inference.load_models()
f = open(glob.glob('data/idr_5000/*.jpg')[0], 'rb').read()
img = inference.preprocess_image(f)
descriptors, color_hist = inference.get_orb_and_color_features(img)
bovw_raw = inference.get_bovw_histogram(descriptors, inference._kmeans_model)
bovw_tfidf = inference._tfidf.transform([bovw_raw]).toarray()[0]
fused = np.hstack((bovw_tfidf, color_hist))
final_features = inference._scaler.transform([fused])

print("try predict_proba:")
try:
    p = inference._svm.predict_proba(final_features)
    print("PROBA WORKED!", np.max(p))
except Exception as e:
    print("PROBA ERROR:", type(e), e)
