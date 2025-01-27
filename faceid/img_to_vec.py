import os
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm

def get_face_analysis():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_images_to_vectors(db_path, output_file="face_vectors.npy"):
    if not os.path.exists(db_path):
        print(f"Error: Directory {db_path} does not exist.")
        return

    detector = get_face_analysis()
    image_files = [f for f in os.listdir(db_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    face_vectors = []
    face_names = []

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(db_path, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            np_image = np.array(image)
            faces = detector.get(np_image)
            for face in faces:
                embedding = face.embedding
                face_vectors.append(embedding)
                face_names.append(image_file)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    np.save(output_file, np.array(face_vectors))
    print(f"Face vectors saved to {output_file}")

    with open("face_names.txt", "w") as f:
        for name in face_names:
            f.write(f"{name}\n")
    print("Face names saved to face_names.txt")

if __name__ == "__main__":
    db_path = "/Users/jamshid/Desktop/image2vec-best-alg/DB_500"
    process_images_to_vectors(db_path)
