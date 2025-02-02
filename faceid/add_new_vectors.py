import os
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm

EXISTING_VECTORS_FILE = "/Users/jamshid/Desktop/image2vec-best-alg/app_v2/vectors/face_vectors.npy"
EXISTING_NAMES_FILE = "/Users/jamshid/Desktop/image2vec-best-alg/app_v2/vectors/face_names.txt"
NEW_IMAGES_DIR = "/Users/jamshid/Desktop/image2vec-best-alg/app_v2/db_test_adding"
OUTPUT_VECTORS_FILE = EXISTING_VECTORS_FILE 
OUTPUT_NAMES_FILE = EXISTING_NAMES_FILE  

def get_face_analysis():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def load_existing_data():
    """Mavjud vektor va ismlar faylini yuklaydi"""
    if os.path.exists(EXISTING_VECTORS_FILE):
        existing_vectors = np.load(EXISTING_VECTORS_FILE)
    else:
        existing_vectors = np.empty((0, 512))  # Bo‘sh massiv (agar mavjud bo‘lmasa)

    if os.path.exists(EXISTING_NAMES_FILE):
        with open(EXISTING_NAMES_FILE, "r") as f:
            existing_names = f.read().splitlines()
    else:
        existing_names = []

    return existing_vectors, existing_names

def process_new_images():
    """Yangi imagelarni vektor holatiga keltiradi"""
    if not os.path.exists(NEW_IMAGES_DIR):
        print(f"Error: Directory {NEW_IMAGES_DIR} does not exist.")
        return None, None

    detector = get_face_analysis()
    image_files = [f for f in os.listdir(NEW_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    new_vectors = []
    new_names = []

    for image_file in tqdm(image_files, desc="Processing new images"):
        image_path = os.path.join(NEW_IMAGES_DIR, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            np_image = np.array(image)
            faces = detector.get(np_image)
            for face in faces:
                embedding = face.embedding
                new_vectors.append(embedding)
                new_names.append(image_file)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    return np.array(new_vectors), new_names

def update_database():
    """Eski va yangi vektorlarni birlashtirib yangilaydi"""
    existing_vectors, existing_names = load_existing_data()
    new_vectors, new_names = process_new_images()

    if new_vectors is None or len(new_vectors) == 0:
        print("No new vectors to add.")
        return

    updated_vectors = np.vstack((existing_vectors, new_vectors))
    np.save(OUTPUT_VECTORS_FILE, updated_vectors)
    print(f"Updated face vectors saved to {OUTPUT_VECTORS_FILE}")

    updated_names = existing_names + new_names
    with open(OUTPUT_NAMES_FILE, "w") as f:
        for name in updated_names:
            f.write(f"{name}\n")
    print(f"Updated face names saved to {OUTPUT_NAMES_FILE}")

if __name__ == "__main__":
    update_database()
