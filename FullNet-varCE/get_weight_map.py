import os
import cv2
import numpy as np


def compute_distances(r, c, boundaries):
    # computes the distances to the nearest and second nearest objects
    # in an image

    min_dists = []
    for b in boundaries:
        b = b[0]
        this_boundary_x = b[:, 0, 0]  # Access the x-coordinates
        this_boundary_y = b[:, 0, 1]  # Access the y-coordinates
        square_distances = (this_boundary_x - r) ** 2 + (this_boundary_y - c) ** 2
        min_dists.append(np.sqrt(np.min(square_distances)))

    sorted_dists = sorted(min_dists)
    d1 = sorted_dists[0]
    d2 = sorted_dists[1]

    return d1, d2


def weight_map():
    data_dir = '/home/data2/MedImg/GlandSeg/CRAG/valid/Annotation/'
    save_dir = '/home/data2/MedImg/GlandSeg/CRAG/valid/weightmaps'
    w0 = 10
    sigma = 5

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_list = os.listdir(data_dir)
    for img_filename in file_list:
        img_path = os.path.join(data_dir, img_filename)
        print(f'Processing image {img_filename}')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        indices = np.unique(img)
        indices = indices[indices != 0]

        m, n = img.shape
        edges = np.zeros((m, n))
        boundaries = [cv2.findContours((img == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                      for i in indices]

        print('=> extracting boundaries...')
        w = np.zeros((m, n))
        if len(boundaries) >= 2:
            for i in range(m):
                if i % 50 == 0:
                    print(f'=> processing rows {i} ~ {i + 50}')
                for j in range(n):
                    d1, d2 = compute_distances(i, j, boundaries)
                    w[i, j] = w0 * np.exp(-(d1 + d2) ** 2 / (2 * sigma ** 2))

        # The weight map is multiplied by 20 to reduce the loss of small values,
        # so they are divided by 20 in the training code.
        # One can also use large bit depth to preserve details.
        save_path = os.path.join(save_dir, f'{img_filename[:-4]}_weight.png')
        cv2.imwrite(save_path, ((w + 1) * 20 / 255.0))


if __name__ == "__main__":
    weight_map()

# import cv2
#
# x = cv2.imread('train_10_weight.png')
# y = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
# t = 2