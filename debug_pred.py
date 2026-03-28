import inference, glob, numpy as np
inference.load_models()
f = open(glob.glob('data/idr_5000/*.jpg')[0], 'rb').read()
res = inference.predict_currency(f)
print(res)
