import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def calcGrayHist(I):
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[int(I[i][j])] += 1
    return grayHist

def equalHist(img):
    h, w = img.shape
    grayHist = calcGrayHist(img)
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


def rotate(image, angle, center=None, scale=1.0):  # 1
    (h, w) = image.shape[:2]  # 2
    if center is None:  # 3
        center = (w // 2, h // 2)  # 4

    M = cv.getRotationMatrix2D(center, angle, scale)  # 5

    rotated = cv.warpAffine(image, M, (w, h))  # 6
    return rotated

def ImageEnhancement(img,name,path,set):

    #直方图均衡化
    equal = equalHist(img)
    # cv.imshow("img", img)
    # cv.imshow("equal",equal)
    cv.imwrite(path+'/equal/'+str(set)+'/'+str(name)+'.png', equal)
    #拉普拉斯算子滤波
    kernel_sharpen_1 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
    laplace = cv.filter2D(img, -1, kernel_sharpen_1)
    cv.imwrite(path + '/laplace/' +str(set)+'/'+ str(name) +'.png', laplace)
    equalLaplace = cv.filter2D(equal, -1, kernel_sharpen_1)
    cv.imwrite(path + '/equal+laplace/'+str(set)+'/' + str(name) + '.png',equalLaplace)
    # output_2 = cv.filter2D(out,-1,kernel_sharpen_2)
    # cv.imshow("output_1", output_1)
    # cv.imshow("output_2", output_2)
    # output_2 = cv.filter2D(img,-1,kernel_sharpen_1)
    # output_2 = cv.filter2D(out,-1,kernel_sharpen_2)
    # cv.imshow("output_2", output_2)

    # 高斯滤波
    # img_Guassian = cv.GaussianBlur(output_1,(5,5),0)

    # 中值滤波
    img_median = cv.medianBlur(equalLaplace, 5)
    cv.imwrite(path + '/equal+laplace+img_median/'+str(set)+'/' + str(name) + '.png', img_median)
    #print(path + '/equal+laplace+img_median/'+str(set)+'/' + str(name) + '-'+str(set)+ '.png')
    #a = rotate(img_median, 180)
    # cv.imshow("img_Guassian", img_Guassian)
    #cv.imshow("img_median", img_median)
    #cv.imshow("a", a)

    #cv.waitKey(0)
    '''
    grayHist = calcGrayHist(img)
    grayHist1 = calcGrayHist(out)
    x = np.arange(256)
    # 绘制灰度直方图
    plt.plot(x, grayHist, 'r', linewidth=2, c='black')
    plt.plot(x, grayHist1, 'r', linewidth=2, c='red')
    plt.xlabel("gray Label")
    plt.ylabel("number of pixels")
    plt.show()'''

train_size=25
test_size=5
path='C://Users/oyy/PycharmProjects/imageEnhancement/image'
for i in range(train_size):
    img = cv.imread("C://Users/oyy/Desktop/dataset/dataset/new train set/train_img/"+str(i)+".png", 0)
    ImageEnhancement(img, i, path,'train')
#for i in range(test_size):
    #img = cv.imread("C://Users/oyy/Desktop/dataset/dataset/new_test_set/test_img/"+str(i)+".png", 0)
    #ImageEnhancement(img, i, path,'test')


