import cv2
import os 
import random

##image dir
folder_path_original = "...folder_path_original"
folder_path_gt = "...folder_path_gt"

images_original = []
images_gt = []

#train, test and validation folder paths
train_path = "...train_path"
val_path = "...val_path"
test_path = "...test_path"

#dimensions
dimensions = (600, 400)

#split images into train, validation and test 
#train 70%, validation 10% and test 20%
train_per = 0.7
val_per = 0.1

#pick random images for training
seed = 42

def store_images(images, path):
    counter = 1
    for image in images:
        name_image= str(counter) + ".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
        counter = counter + 1


def generate_image_tuples(folder_path_original, folder_path_gt, dimensions):
#load original and ground truth images
    image_names_gt = os.listdir(folder_path_gt)
    image_names_original = os.listdir(folder_path_original)
    images = []

    for image_name_gt in os.listdir(folder_path_gt):
        # check image extension
        if os.path.splitext(image_name_gt)[1] in [".jpg"] and image_name_gt in image_names_gt:

            related_images = [x for x in image_names_original if image_name_gt.split(".")[0] in x]

            original_images = []
            for image_name in related_images:
                image_path_original = os.path.join(folder_path_original, image_name)
                image_original = cv2.imread(image_path_original)
                # normalize original image
                image_original = cv2.resize(image_original, dimensions)
                original_images.append(image_original)

            image_path_gt = os.path.join(folder_path_gt, image_name_gt)
            image_gt = cv2.imread(image_path_gt)
            # normalize gt image
            image_gt = cv2.resize(image_gt, dimensions)
            for image in original_images:
                image_concat = cv2.hconcat([image, image_gt])
                images.append(image_concat)
    return images


def generate_random_numbers(seed, set_len, images_len):
    random.seed(seed)
    random_numbers = []
    while len(random_numbers) < set_len:
        random_number = random.randint(0, images_len - 1)
        if random_number not in random_numbers:
            random_numbers.append(random_number)
    return random_numbers


def main():
    images = generate_image_tuples(folder_path_original, folder_path_gt, dimensions)
    images_len = len(images)

    train_len = round(len(images)*train_per)
    val_len = round(len(images)*val_per)
    test_len = len(images) - (val_len + train_len)

    random_numbers_train = generate_random_numbers(seed, train_len, images_len)
    random_numbers_validation = generate_random_numbers(seed, val_len, images_len)
    random_numbers_test = generate_random_numbers(seed, test_len, images_len)

    #take images from random numbers
    images_train = []
    images_val = []
    images_test = []

    for index in random_numbers_train:
        images_train.append(images[index])

    for index in random_numbers_validation:
        images_val.append(images[index])

    for index in random_numbers_test:
        images_test.append(images[index])

    #storage images
    if(not os.path.isdir(train_path)):
        os.makedirs(train_path)
    store_images(images_train, train_path)

    if(not os.path.isdir(val_path)):
        os.makedirs(val_path)
    store_images(images_val, val_path)

    if(not os.path.isdir(test_path)):
        os.makedirs(test_path)
    store_images(images_test, test_path)

    print("train: ", len(images_train))
    print("val: ",len(images_val))
    print("test: ", len(images_test))

if __name__ == "__main__":
    main()
