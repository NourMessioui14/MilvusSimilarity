import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Connect to Milvus and set up the collection
def connect_to_milvus():
    connections.connect(alias="default", host='localhost', port="19530", timeout=30)
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="words", dtype=DataType.VARCHAR, max_length=255)
    ]
    schema = CollectionSchema(fields, description="Image similarity search")
    collection_name = "image_similarity"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    collection = Collection(name=collection_name, schema=schema)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 256}}
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

# Load the pre-trained ResNet model
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
    model.eval()
    return model

# Preprocess the image and extract the embedding
def preprocess_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    # Normalize the embedding to ensure consistent distance calculations
    normalized_embedding = embedding / np.linalg.norm(embedding)
    return normalized_embedding

# Compute and insert embeddings into Milvus
def insert_embeddings(collection, model, image_folder, batch_size=200):  # Larger batch size for more efficient insertion
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    embeddings = []
    words = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=12) as executor:  # Increase the number of workers for better parallelism
        future_to_path = {executor.submit(preprocess_image, path, model): path for path in image_paths}
        for i, future in enumerate(as_completed(future_to_path)):
            embedding = future.result()
            if embedding is not None:
                embeddings.append(embedding)
                words.append(os.path.basename(future_to_path[future]).split('.')[0])  # Use the image filename (without extension) as the word
            if len(embeddings) >= batch_size:
                # Insert data as two lists: embeddings and words
                collection.insert([embeddings, words])
                embeddings = []
                words = []
                print(f"Batch {i // batch_size} inserted.")
    if embeddings:
        # Insert remaining data
        collection.insert([embeddings, words])
    collection.flush()
    print(f"Total time for insertion: {time.time() - start_time} seconds")

# Search for similar images in Milvus
def search_similar_images(collection, query_image_path, model, top_k=5):
    collection.load()
    query_embedding = preprocess_image(query_image_path, model)
    if query_embedding is not None:
        search_params = {"metric_type": "L2"}
        results = collection.search(
            data=[query_embedding], 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k,
            output_fields=["words"]
        )
        return results[0]
    else:
        print("Query image could not be processed.")
        return []

# Select an image file
def select_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    return file_path

# Display the similar images using Matplotlib
def display_similar_images(results, image_folder):
    num_images = len(results)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for ax, result in zip(axes, results):
        image_path = os.path.join(image_folder, f"{result.entity.get('words')}.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            ax.imshow(image)
            ax.set_title(f"Distance: {result.distance:.2f}")
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Image not found', horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis('off')
    plt.show()

def main():
    collection = connect_to_milvus()
    model = load_model()
    image_folder = "/home/nour/MilvusSimilarity/db_teeth_"
    insert_embeddings(collection, model, image_folder)
    query_image = select_image()
    if query_image:
        results = search_similar_images(collection, query_image, model)
        if results:
            display_similar_images(results, image_folder)
        else:
            print("No similar images found.")
    else:
        print("No image selected.")

if __name__ == "__main__":
    main()
