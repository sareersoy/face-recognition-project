import cv2
import numpy as np
import os
import logging
import pickle
import copy

def save_pkl(path, arr):
    with open(path, 'wb') as f:
        pickle.dump(arr, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

def haar_cascade_face_detection(images, labels, output_dir, faces_npy_path):

    if os.path.exists(faces_npy_path):
        logging.info(f"Loading saved {faces_npy_path}")    
        return load_pkl(faces_npy_path)

    logging.info("Running haar cascade face detection")
    
    all_faces = []

    # detect faces
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for i, (image, name) in enumerate(zip(images, labels)):
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        all_faces.append(faces)        
        
        # Create a new directory for each person

        if output_dir:

            directory = f"{output_dir}/{name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
                current_count = 0

            faces_drawn_image = copy.deepcopy(image)
            for (x, y, w, h) in faces:
                cv2.rectangle(faces_drawn_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            faces_drawn_image = cv2.cvtColor(faces_drawn_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(os.path.join(directory, f"{current_count}.jpg"), faces_drawn_image)
            current_count += 1

    logging.info("Haar cascade face detection completed")
    
    if faces_npy_path:
        save_pkl(faces_npy_path, all_faces)
        logging.info(f"Saved {faces_npy_path}")

    return all_faces

def face_selection(images, labels, all_faces, output_dir, selected_faces_npy_path, unidentified_idxs_npy_path):

    if os.path.exists(selected_faces_npy_path) and os.path.exists(unidentified_idxs_npy_path):
        logging.info(f"Loading saved {selected_faces_npy_path}")
        selected_faces = load_pkl(selected_faces_npy_path)
        
        logging.info(f"Loading saved {unidentified_idxs_npy_path}")
        unidentified_idxs = load_pkl(unidentified_idxs_npy_path)
        
        return selected_faces, unidentified_idxs
    
    logging.info(f"Selecting faces from extracted faces")

    selected_faces = []
    unidentified_idxs = []

    for i, (image, name, faces) in enumerate(zip(images, labels, all_faces)):
        if len(faces) == 0:
            logging.debug(f'No face has been detected on image {name}')
            unidentified_idxs.append(i)

        else:
            # Find the largest face
            largest_face_idx = np.argmax(faces[:, 2] * faces[:, 3])
            (x, y, w, h) = faces[largest_face_idx]

            selected_faces.append([x, y, w, h])

            # Crop the largest face region
            cropped_face = image[y:y+h, x:x+w]

            if output_dir:
                directory = f"{output_dir}/{name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    current_count = 0

                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                cv2.imwrite(os.path.join(directory, f"{current_count}.jpg"), cropped_face)
                current_count += 1
    
    logging.info(f"Face selection completed, total of {len(selected_faces)} faces has been selected")
    
    if selected_faces_npy_path and unidentified_idxs_npy_path:
        save_pkl(selected_faces_npy_path, selected_faces)
        logging.info(f"Saved {selected_faces_npy_path}")

        save_pkl(unidentified_idxs_npy_path, unidentified_idxs)
        logging.info(f"Saved {unidentified_idxs_npy_path}")
    
    return selected_faces, unidentified_idxs


