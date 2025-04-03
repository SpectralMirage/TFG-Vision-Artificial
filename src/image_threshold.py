import cv2
import os
import numpy as np
import random

path_snippets = "...path_snippets"
path_images = "...path_images"
path_dest = "..path_dest"

dimensions_blank = (400, 600)
dimensions_snipped = (50, 20)

def create_snipped_images(image_path):
    # load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (T, thresh) = cv2.threshold(gray, 231, 255, cv2.THRESH_BINARY)

    # normalize
    snipped = cv2.resize(thresh, dimensions_snipped)

    # create blank image
    blank_image = np.tile(255, (400, 600, 3))

    # get height and  width
    h_snipped, w_snipped = snipped.shape[:2]
    h_blank, w_blank = blank_image.shape[:2]

    pos_x1 = random.randint(0, (w_blank - 1) - w_snipped)
    pos_y1 = random.randint(0, (h_blank - 1) - h_snipped)

    pos_x2 = pos_x1 + w_snipped
    pos_y2 = pos_y1 + h_snipped

    # replace location with snipped image
    blank_image[pos_y1:pos_y2, pos_x1:pos_x2, 0] = snipped
    blank_image[pos_y1:pos_y2, pos_x1:pos_x2, 1] = snipped
    blank_image[pos_y1:pos_y2, pos_x1:pos_x2, 2] = snipped
    blank_image = blank_image.astype(np.uint8)

    #add alpha channel
    alpha_channel = cv2.bitwise_not(blank_image[:, :, 0])
    alpha_channel = alpha_channel.astype(np.uint8)
    blank_image_alpha = cv2.merge((blank_image[:, :, 0], blank_image[:, :, 1], blank_image[:, :, 2], alpha_channel))

    blank_image_alpha = cv2.convertScaleAbs(blank_image_alpha, alpha=160)
    return blank_image_alpha


def combine_images(image, snipped):
    image = cv2.resize(image, (600, 400))
    image_copy = np.copy(image)
    alpha_channel = np.ones((image_copy.shape[0], image_copy.shape[1]), dtype=np.uint8) * 255  # Inicialmente todos los píxeles son opacos (valor 255)
    image_copy = cv2.merge((image_copy[:, :, 0], image_copy[:, :, 1], image_copy[:, :, 2], alpha_channel))

    combined_image = cv2.bitwise_and(snipped, image_copy)
    return combined_image


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_names = os.listdir(path_images)
    for image_name in image_names:
        path_image = os.path.join(path_images, image_name)
        
        image = cv2.imread(path_image)
        snipped_names = os.listdir(path_snippets)
        combined_images = []
        for name in snipped_names:
            snipped_path = os.path.join(path_snippets, name)
            snipped = create_snipped_images(snipped_path)
            combined_images.append(combine_images(image, snipped))

        #stored images
        if(not os.path.isDir(path_dest)):
           os.makedirs(path_dest)
        os.chdir(path_dest)
        image_name = image_name.split(".")[0]
        for i in range(len(combined_images)):
            index = "_" + str((i + 1)) + ".jpg"
            name = image_name + index
            cv2.imwrite(name, combined_images[i])

if __name__ == "__main__":
    main()
