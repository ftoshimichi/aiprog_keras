import numpy as np

img = np.array([[2,1,2,1],
                [1,2,3,2],
                [2,3,1,3],
                [2,0,1,1]])

filter = np.array([[1,3,2], [1,2,0], [2,3,1]])

img_width = img.shape[0]
img_height = img.shape[1]

filter_width = filter.shape[0]
filter_height = filter.shape[1]

for i in range(img_height - filter_height + 1):
    for j in range(img_width - filter_width + 1):
        z = 0
        z += img[i][j + 0] * filter[0][0]
        z += img[i][j + 1] * filter[0][1]
        z += img[i][j + 2] * filter[0][2]

        z += img[i + 1][j + 0] * filter[1][0]
        z += img[i + 1][j + 1] * filter[1][1]
        z += img[i + 1][j + 2] * filter[1][2]

        z += img[i + 2][j + 0] * filter[2][0]
        z += img[i + 2][j + 1] * filter[2][1]
        z += img[i + 2][j + 2] * filter[2][2]

        print(z, end=' ')
    print("")
