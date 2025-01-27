import numpy as np
import os
from PIL import Image
from insightface.app import FaceAnalysis

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def find_similar_face(image_path, embeddings_path):
    # Initialize face detector
    detector = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    detector.prepare(ctx_id=0, det_size=(640, 640))
    
    # Process input image
    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image)
    faces = detector.get(np_image)
    
    if not faces:
        raise ValueError("No face detected in the input image")
    
    img_embedding = faces[0].get('embedding')
    img_embedding = np.array(img_embedding)
    
    # Load existing embeddings
    all_embeddings = np.load(embeddings_path)
    
    # Read face names
    with open('face_names.txt', 'r') as f:
        face_names = f.read().splitlines()
    
    # Calculate distances
    distances = []
    for embedding in all_embeddings:
        distance = findEuclideanDistance(img_embedding, embedding)
        distances.append(distance)
    
    # Find most similar face
    min_distance_idx = np.argmin(distances)
    similar_face_name = face_names[min_distance_idx]
    
    return similar_face_name, distances[min_distance_idx]

# Test the function
image_path = "test.jpg"
embeddings_path = "face_vectors.npy"
similar_name, distance = find_similar_face(image_path, embeddings_path)
print(f"Eng o'xshash shaxs: {similar_name}")
print(f"O'xshashlik darajasi: {1-distance:.2%}")