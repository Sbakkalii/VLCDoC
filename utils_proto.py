import os

import cv2
import numpy as np
import tensorflow as tf
from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
image_size = 224


def Preprocess_Image(image):
    image = cv2.imread(image)[:,:, ::-1]
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255
    return image

def prepare_image_data(path):
    data = path
    classes = (os.listdir(path))
    paths_train = [os.path.join(data, o) for o in os.listdir(data) if os.path.isdir(os.path.join(data, o))]
    images = []
    texts = []
    labels = []
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                data_path = "/path/to/data/"
                #split_path = os.path.join(data_path, path.split(os.path.sep)[-1])
                # split_path = os.path.splitext(split_path)[0]
                images.append(os.path.join(data_path, "Image/Train_Data/", path.split(os.path.sep)[-2], os.path.splitext(path.split(os.path.sep)[-1])[0]+ ".jpg"))
                texts.append(os.path.join(data_path, "Text/Train_Data/", path.split(os.path.sep)[-2], os.path.splitext(path.split(os.path.sep)[-1])[0]+ ".txt"))
                labels.append(root.split(os.path.sep)[-1])
    return images, texts, labels


# Return image class based on list entry (path)
def getClass(img):
    return img.split(os.path.sep)[-2]


def process_image_data(path, image_data, text_data):
    classes = (os.listdir(path))
    img = []
    txt = []
    label = []
    lb = LabelEncoder()
    for image, text in list(zip(image_data[0: len(image_data)], text_data[0: len(text_data)])):
        img.append(image)
        txt.append(text)
        label.append(getClass(image))
    lb.fit(list(classes))
    label = lb.transform(np.array(label))
    label = np.array(label, dtype="float32")
    return img, txt, label


def prepare_text_data(path):
    data = path
    classes = (os.listdir(data))
    paths_train = [os.path.join(data, o) for o in os.listdir(data) if os.path.isdir(os.path.join(data, o))]
    texts = []
    labels = []
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                texts.append(path)
                labels.append(root.split(os.path.sep)[-1])
    return texts, labels


def process_text_data(path, data):
    classes = (os.listdir(path))
    img = []
    label = []
    lb = LabelEncoder()
    for i in data[0: len(data)]:
        img.append(i)
        label.append(getClass(i))
    lb.fit(list(classes))
    label = lb.transform(np.array(label))
    label = np.array(label, dtype="float32")
    return img, label


def Preprocess_Text(text_file, tokenizer, max_seq_len):
    with open(text_file, 'r', encoding="utf-8", errors='ignore') as reader:
        text = reader.read()
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    cut_point = min(len(token_ids), max_seq_len - 2)
    input_ids = token_ids[:cut_point]
    input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
    return input_ids


def create_training_rvlcdip_dataset(image_path_train, text_path_train):
    #  Load Image Data and Labels
    train_image_set, data_train_text, train_image_labels = prepare_image_data(image_path_train)
    train_image_data, data_train_text, train_image_label = process_image_data(image_path_train, train_image_set,
                                                                              data_train_text)

    # Prepare Multimodel Data
    data = list(zip(train_image_data, data_train_text))
    labels = train_image_label

    data = shuffle(data, random_state=len(data))
    labels = shuffle(labels, random_state=len(labels))

    return data


def create_test_dataset(buffer_size, image_path_test, text_path_test):
    #############  Load Image Data and Labels  ####################################
    test_image_set, test_image_label = prepare_image_data(path=image_path_test)
    test_image_data, test_image_label = process_image_data(path=image_path_test, data=test_image_set)

    #############  Load Text Data and Labels  ####################################

    data_test_text, labels_test_text = prepare_text_data(path=text_path_test)
    data_test_text, labels_test_text = process_text_data(path=text_path_test, data=data_test_text)

    #############  Prepare Multimodel Data  ####################################
    data_test = list(zip(test_image_data, data_test_text))
    data_test = shuffle(data_test, random_state=buffer_size)

    return data_test