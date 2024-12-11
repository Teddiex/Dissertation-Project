import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from shutil import copyfile
import pandas as pd
import numpy as np



DATASET_PATH = 'dataset'

def read_labels_from_file(file_path):
    labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 4:
                patient_id, filename, label, data_source = parts[:4]
                labels[filename] = label
    return labels

def combine_images_and_labels(root_folder, output_folder):
    all_images = []
    all_labels = []

    #for folder_name in ['test', 'train', 'val']:
    for folder_name in ['train', 'val']:
        folder_path = os.path.join(root_folder, folder_name)
        label_file_path = os.path.join(root_folder, f'{folder_name}.txt')

        current_labels = read_labels_from_file(label_file_path)

        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):  
                image_path = os.path.join(folder_path, file_name)
                all_images.append(image_path)
                all_labels.append(current_labels.get(file_name, 'unknown'))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Write the images and labels to the output folder
    for i, (image_path, label) in enumerate(zip(all_images, all_labels)):
        # Copy the image to the output folder
        image_name = f'image_{i}.jpg'
        output_image_path = os.path.join(output_folder, image_name)
        copyfile(image_path, output_image_path)

        # Write the filename and label to the .txt file
        txt_file_path = os.path.join(output_folder, 'labels.txt')
        with open(txt_file_path, 'a') as txt_file:
            txt_file.write(f'{image_name} {label}\n')

    return all_images, all_labels

def encode_labels(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    encoded_lines = []
    for line in lines:
        image_name, label = line.strip().split()
        # Encode 'positive' as 1 and 'negative' as 0
        encoded_label = '1' if label.lower() == 'positive' else '0'
        encoded_lines.append(f'{image_name} {encoded_label}\n')

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(encoded_lines)

class COVIDDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.annotations = pd.read_csv(txt_file, header=None, sep = ' ')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx): 
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



