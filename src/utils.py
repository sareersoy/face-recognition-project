import cv2
import os
from sklearn.model_selection import GroupShuffleSplit


def filter_with_idxs_inplace(source_arr, filter):
    rev_sorted_idxs = sorted(filter, reverse=True)
    for idx in rev_sorted_idxs:
        del source_arr[idx]

def crop_images_in_place(images, bounding_boxes):
    for i in range(len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i]
        images[i] = images[i][y:y+h, x:x+w]

def resize_images_in_place(images, new_size):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], new_size, interpolation = cv2.INTER_AREA)


def save_images(images, labels, root_dir):
    
    if os.path.exists(root_dir):
        return
    
    for _, (image, label) in enumerate(zip(images, labels)):
        label_path = os.path.join(root_dir, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
            counter = 0

        img_path = os.path.join(label_path, f"{counter}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        counter += 1

def split_dataset(X, y, test_split_size, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_split_size, random_state=random_state)
    train_idxs, test_idxs = next(splitter.split(X, y, groups=y))

    X_train, X_test = X[train_idxs], X[test_idxs]
    y_train, y_test = y[train_idxs], y[test_idxs]

    return X_train, X_test, y_train, y_test

