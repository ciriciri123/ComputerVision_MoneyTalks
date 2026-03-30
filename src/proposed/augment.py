import cv2
import numpy as np
import os
import glob

# Tentukan lokasi folder dataset-mu (Sesuaikan jika berbeda)
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def augment_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    # Dapatkan nama file asli dan ekstensinya
    dir_name = os.path.dirname(img_path)
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    
    # 1. BIKIN GELAP (Dark)
    dark_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0) # Turunkan kecerahan 50%
    cv2.imwrite(os.path.join(dir_name, f"{name}_dark{ext}"), dark_img)
    
    # 2. BIKIN SILAU (Bright)
    bright_img = cv2.convertScaleAbs(img, alpha=1.2, beta=40) # Naikkan kecerahan
    cv2.imwrite(os.path.join(dir_name, f"{name}_bright{ext}"), bright_img)
    
    # 3. BIKIN BLUR / BURAM (Motion Blur)
    blur_img = cv2.GaussianBlur(img, (11, 11), 0)
    cv2.imwrite(os.path.join(dir_name, f"{name}_blur{ext}"), blur_img)
    
    # 4. BIKIN NOISE (Bintik-bintik kamera murah)
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noise_img = cv2.add(img, noise)
    cv2.imwrite(os.path.join(dir_name, f"{name}_noise{ext}"), noise_img)

def run_augmentation():
    valid_classes = ['idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 'idr_20000', 'idr_50000', 'idr_100000']
    
    print(f"[*] Memulai Proses Data Augmentation di folder: {DATASET_DIR}")
    
    total_augmented = 0
    for label in valid_classes:
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue
            
        # Cari gambar asli (Hanya yang TIDAK memiliki akhiran _dark, _blur, dll)
        # Ini untuk mencegah augmentasi dilakukan berulang-ulang pada gambar yang sudah di-augment
        all_images = glob.glob(os.path.join(class_dir, "*.*"))
        original_images = [img for img in all_images if not any(x in img for x in ['_dark', '_bright', '_blur', '_noise'])]
        
        print(f" -> Memproses {len(original_images)} gambar asli di '{label}'...")
        
        for img_path in original_images:
            augment_image(img_path)
            total_augmented += 4 # 4 variasi per gambar
            
    print(f"\n[*] Selesai! Berhasil menciptakan {total_augmented} gambar simulasi dunia nyata baru.")

if __name__ == "__main__":
    run_augmentation()