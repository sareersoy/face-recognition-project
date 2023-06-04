import tarfile
import os
import numpy as np
from PIL import Image
import logging

def extract_lfw_dataset(lfw_tar_path, lfw_extract_path, lfw_root_path):
    
    if os.path.exists(lfw_root_path):
        logging.info('Dataset already extracted')
        return
    
    logging.info('Extracting dataset')

    try:
        tar = tarfile.open(lfw_tar_path, 'r:gz')
        tar.extractall(lfw_extract_path)
        tar.close()
    
    except Exception as e:
        logging.error("Dataset could not be extracted with error:", e)
    
    logging.info('Extracted dataset')

def load_lfw_dataset(lfw_extract_path):

    logging.info('Loading lfw dataset into memory')

    images = []
    labels = []
    
    person_dirs = os.listdir(lfw_extract_path)
    for person_dir in person_dirs:
        person_path = os.path.join(lfw_extract_path, person_dir)
        image_files = os.listdir(person_path)
        
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            image = Image.open(image_path)
            image = np.array(image)

            images.append(image)
            labels.append(person_dir)

    logging.info('Loaded lfw dataset into memory')

    return images, labels
