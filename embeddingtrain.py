from deepface import DeepFace
import os
import pickle

def get_embeddings(dataset_path, embedding_file="face_embeddings.pkl"):
    embeddings = []
    labels = []

    # Walk through the dataset directory
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for image in os.listdir(person_path):
                image_path = os.path.join(person_path, image)

                try:
                    # Generate embedding using FaceNet
                    embedding = DeepFace.represent(
                        img_path=image_path, 
                        model_name="Facenet", 
                        enforce_detection=False
                    )[0]['embedding']
                    embeddings.append(embedding)
                    labels.append(person)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # Save embeddings to a file
    with open(embedding_file, "wb") as file:
        pickle.dump({"embeddings": embeddings, "labels": labels}, file)
    print(f"Embeddings saved to {embedding_file}")

# Run embedding extraction
get_embeddings("./dataset/known_faces")  # Corrected to 'dataset'