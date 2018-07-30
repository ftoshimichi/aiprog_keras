import numpy as np

img = np.array([[2,1,2,1],
                [1,2,3,2],
                [2,3,1,3],
                [2,0,1,1]])

img_width = img.shape[0]
img_height = img.shape[1]

pool_width = 2
pool_height = 2

for i in range(0, img_width, pool_width):
    for j in range(0, img_height, pool_height):
        pool = [img[i][j + 0], img[i][j+ 1], img[i + 1][j + 0], img[i + 1][j + 1]]
        print(np.max(pool), end=' ')
    print("")
