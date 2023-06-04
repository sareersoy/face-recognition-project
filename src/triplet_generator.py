from enum import Enum
import numpy as np
import random 
import logging

class TripletMiningStrategy(Enum):
    RANDOM = 1
    HARD = 2

import numpy as np

def get_batch(X, y, batch_size):
    idxs = np.random.choice(X.shape[0], size=batch_size)
    return X[idxs], y[idxs]

def create_label_to_images_dict(X, y):
    label_to_images = {}
    for img, label in zip(X, y):
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(img)
    
    return label_to_images

def compute_distance(a, b):
    return np.linalg.norm(a - b)

def compute_distance_matrix(X):
    num_images = len(X)
    distance_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i+1, num_images):  # j>i to avoid redundant computations
            distance_matrix[i][j] = compute_distance(X[i], X[j])
            distance_matrix[j][i] = distance_matrix[i][j]  # The matrix is symmetric
    return distance_matrix


def find_closest_diff_label(anchor_index, labels, dist_matrix):
    # Get the label of the anchor image
    anchor_label = labels[anchor_index]

    # Create a mask where true means the label is different from the anchor
    diff_label_mask = labels != anchor_label

    # Set the distances of the images with the same label to be infinity
    dist_matrix = dist_matrix.copy()
    dist_matrix[:, ~diff_label_mask] = np.inf

    # Get the index of the image with the smallest non-infinite distance to the anchor
    closest_image_index = np.argmin(dist_matrix[anchor_index])

    return closest_image_index

class TripletGenerator():
    def __init__(self, X, y, batch_size=128, strategy=TripletMiningStrategy.RANDOM):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.strategy = strategy


    def next_batch(self):
        if self.strategy == TripletMiningStrategy.RANDOM:
            return self._next_random_batch()
        
        elif self.strategy == TripletMiningStrategy.HARD:
            return self._next_hard_batch()
    

    def _next_random_batch(self):
        # sample the batch
        while True:
            X_batch, y_batch = get_batch(self.X, self.y, self.batch_size)
            
            # get reqirements
            labels_to_images = self._get_label_to_images(X_batch, y_batch)
            labels_with_at_least_two_images = self._get_valid_labels(labels_to_images)
            triplets_for_current_batch = []
            
            if len(labels_with_at_least_two_images) > 0:
                break


        for _ in range(self.batch_size):

            # randomly choose anchor and positive from labels that have at least 2 images
            anchor_label = random.choice(labels_with_at_least_two_images)
            anchor, positive = random.sample(labels_to_images[anchor_label], 2)

            # randomly choose negative label with only condition that it's not same as anchor label
            negative_label = random.choice([label for label in y_batch if label != anchor_label])
            negative = random.choice(labels_to_images[negative_label])
            triplets_for_current_batch.append([anchor, positive, negative])
        
        return np.array(triplets_for_current_batch)

    def _next_hard_batch(self):
        
        while True:
            X_batch, y_batch = get_batch(self.X, self.y, self.batch_size)
            
            # get reqirements
            labels_to_images = self._get_label_to_images(X_batch, y_batch)
            labels_with_at_least_two_images = self._get_valid_labels(labels_to_images)
            distance_matrix = compute_distance_matrix(X_batch)
            triplets_for_current_batch = []
            
            if len(labels_with_at_least_two_images) > 0:
                break


        for _ in range(self.batch_size):

            # randomly choose anchor and positive from labels that have at least 2 images
            anchor_label = random.choice(labels_with_at_least_two_images)
            anchor_class_size = len(labels_to_images[anchor_label])
            anchor_idx, positive_idx = random.sample(list(range(anchor_class_size)), 2)

            anchor = labels_to_images[anchor_label][anchor_idx]
            positive = labels_to_images[anchor_label][positive_idx]
            
            negative_idx = find_closest_diff_label(anchor_idx, y_batch, distance_matrix)

            negative = X_batch[negative_idx]
            triplets_for_current_batch.append([anchor, positive, negative])

        return triplets_for_current_batch


    def _get_valid_labels(self, label_to_images):
        return [label for label, images in label_to_images.items() if len(images) >= 2]


    def _get_label_to_images(self, X_batch, y_batch):
        return create_label_to_images_dict(X_batch, y_batch)
    
    