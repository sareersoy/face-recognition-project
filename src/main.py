# SET UP LOGGING
import warnings

import sys


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IMPORTS

import numpy as np
import copy

# EXTRACT AND LOAD THE DATASET

from data_loader import extract_lfw_dataset, load_lfw_dataset

dataset_version = 'lfw-deepfunneled'

LFW_TAR_PATH = f"./data/{dataset_version}.tgz"
LFW_EXTRACT_PATH = "./data/"
LFW_ROOT_PATH = f"./data/{dataset_version}"

extract_lfw_dataset(LFW_TAR_PATH, LFW_EXTRACT_PATH, LFW_ROOT_PATH)
images, labels = load_lfw_dataset(LFW_ROOT_PATH)

# PREPROCESS THE DATASET

from preprocess import haar_cascade_face_detection, face_selection

all_faces = haar_cascade_face_detection(images, labels, 
                                        output_dir="./data/haar_faces/",
                                        faces_npy_path="./saves/all_faces.pkl")

selected_faces, unidentified_idxs = face_selection(images, labels, all_faces,
                                        output_dir="./data/selected_faces",
                                        selected_faces_npy_path="./saves/selected_faces.pkl",
                                        unidentified_idxs_npy_path="./saves/unidentified_idxs.pkl")

# UPDATE IMAGES AND LABELS

from utils import filter_with_idxs_inplace
filter_with_idxs_inplace(labels, filter=unidentified_idxs)
filter_with_idxs_inplace(images, filter=unidentified_idxs)

# CROP AND RESIZE IMAGES

from utils import crop_images_in_place, resize_images_in_place

crop_images_in_place(images, bounding_boxes=selected_faces)
resize_images_in_place(images, new_size=(128, 128))

# SAVE IMAGES
#from utils import save_images

#save_images(images, labels, root_dir="./data/resized_images")
#del images; del labels

# CONVERT TO NUMPY ARRAY

X = np.array(images); del images
y = np.array(labels); del labels

# SPLIT DATASET
from utils import split_dataset

X_train, X_test, y_train, y_test = split_dataset(X, y, test_split_size=0.2, random_state=10)

### TRAINING ###
from sklearn import metrics

def compute_metrics(model, test_triplets, threshold=0.5):
    # Reshape test_triplets to the format the model expects
    test_triplets = np.transpose(test_triplets, (1, 0, 2, 3, 4))

    # normalize
    test_triplets = test_triplets.astype(np.float32) / 255

    # Compute the distances using the model
    distances_positive, distances_negative = model.predict([*test_triplets])

    # calucate metrics 

    TP = np.sum(distances_positive < threshold)
    FN = np.sum(distances_positive >= threshold)
    TN = np.sum(distances_negative > threshold)
    FP = np.sum(distances_negative <= threshold)

    return TP, FN, TN, FP


from triplet_generator import TripletGenerator, TripletMiningStrategy


LEARNING_RATE = 0.01
EPOCHS=3
TRAIN_BATCH_SIZE=512
TEST_BATCH_SIZE=X_test.shape[0]
TRIPLET_MINING_STRATEGY = TripletMiningStrategy.RANDOM



from keras.optimizers import Adam
from model import get_siamese_network, SiameseModel

# CONSTRUCT MODEL

siamese_network = get_siamese_network()
siamese_model = SiameseModel(siamese_network, margin=1)
optimizer = Adam(learning_rate=LEARNING_RATE)#,epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)




train_losses = []


test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []


train_triplet_generator = TripletGenerator(X_train, y_train, TRAIN_BATCH_SIZE, TRIPLET_MINING_STRATEGY)
test_triplet_generator = TripletGenerator(X_test, y_test, TEST_BATCH_SIZE, TRIPLET_MINING_STRATEGY)


for epoch in range(EPOCHS):

    # training steps

    epoch_losses = []
    num_steps_per_epoch = X_train.shape[0] // TRAIN_BATCH_SIZE
    for step in range(num_steps_per_epoch):
        triplets = train_triplet_generator.next_batch()

        # convert from (3, #batch, height, width, channels) to (#batch, 3, height, width, channels)
        triplets_corrected = np.transpose(triplets, (1, 0, 2, 3, 4)) 
        triplets_corrected = triplets_corrected.astype(np.float32) / 255
        
        loss = siamese_model.train_on_batch([*triplets_corrected])
        
        epoch_losses.append(loss)

    epoch_mean_train_loss = np.mean(epoch_losses)
    train_losses.append(epoch_mean_train_loss)
    

    # testing steps
    num_test_steps_per_epoch = X_test.shape[0] // TEST_BATCH_SIZE
    
    accuracies, precisions, recalls, f1s = [],[],[],[]
    tp_rates, fn_rates, tn_rates, fp_rates = [], [], [], []

    for test_step in range(num_test_steps_per_epoch):
        test_triplets = test_triplet_generator.next_batch()
        tp, fn, tn, fp = compute_metrics(siamese_model, test_triplets)  
        
        tp_rates.append(tp/TEST_BATCH_SIZE)
        fn_rates.append(fn/TEST_BATCH_SIZE)
        tn_rates.append(tn/TEST_BATCH_SIZE)
        fp_rates.append(fp/TEST_BATCH_SIZE)
        
        accuracies.append((tp+tn)/(tp + fn + tn + fp))

        precision = tp / (tp + fp)
        precisions.append(precision)

        recall = tp / (tp + fn)
        recalls.append(recall)

        f1 = (2*precision*recall) / (precision + recall)
        f1s.append(f1)
        #accuracies.append((tp+tn)/(2*TEST_BATCH_SIZE))
        
        
    test_accuracy.append(np.mean(accuracies))
    test_precision.append(np.mean(precisions))
    test_recall.append(np.mean(recalls))
    test_f1.append(np.mean(f1s))



    print(f"""
    ==== Epoch {epoch} ====
        avg train loss: {epoch_mean_train_loss}

        avg test tp rate: {np.mean(tp_rates)}
        avg test fn rate: {np.mean(fn_rates)}
        avg test tn rate: {np.mean(tn_rates)}
        avg test fp rate: {np.mean(fp_rates)}
        
        avg test accuracy: {np.mean(accuracies)}
        avg test precision: {np.mean(precisions)}
        avg test recall: {np.mean(recalls)}
        avg test f1: {np.mean(f1s)}


    """)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_losses)
plt.title('Train Loss over Epochs')
plt.xticks([i for i in range(EPOCHS)])
plt.savefig('train_loss.png')

plt.figure()
plt.plot(test_accuracy, label='accuracy')
plt.plot(test_precision, label='precision')
plt.plot(test_recall, label='recall')
plt.plot(test_f1, label='f1 score')
plt.title('Test Metrics over Epochs')
plt.legend()
plt.xticks([i for i in range(EPOCHS)])
plt.savefig('test_metrics.png')

