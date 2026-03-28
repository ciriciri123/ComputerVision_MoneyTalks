import os, glob
from inference import predict_currency
data_dir = r'c:\Kuliah\Semester 4\Final Project\money-talks\data'
for d in os.listdir(data_dir):
  if not d.startswith('idr_'): continue
  p = os.path.join(data_dir, d)
  imgs = glob.glob(os.path.join(p, '*.jpg'))
  if not imgs: continue
  res = predict_currency(open(imgs[0], 'rb').read())
  print(f'True: {d} | Pred: {res}')
