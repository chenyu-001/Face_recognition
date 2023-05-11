import cv2
import matplotlib.pyplot as plt
import numpy as np


def smoothingfilter(image_path):
    # 读取图像
    img = cv2.imread(image_path, 0)

    # 应用均值滤波器
    mean_filter = cv2.blur(img, (5, 5))

    # 应用中值滤波器
    median_filter = cv2.medianBlur(img, 5)

    # 应用高斯滤波器
    gaussian_filter = cv2.GaussianBlur(img, (5, 5), 0)

    # 绘制图像以进行比较
    plt.figure(figsize=(12, 12))

    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Noisy Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(mean_filter, 'gray'), plt.title('Mean Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(median_filter, 'gray'), plt.title('Median Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(gaussian_filter, 'gray'), plt.title('Gaussian Filter')
    plt.xticks([]), plt.yticks([])

    plt.show()

    return mean_filter, median_filter, gaussian_filter


def imageSharpening(img_path):
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 应用拉普拉斯滤波器
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # 由于拉普拉斯滤波器会得到正值和负值，我们需要将其转换为8位无符号整数格式
    laplacian = cv2.convertScaleAbs(laplacian)

    # 锐化图像 = 原图像 + 拉普拉斯
    sharpened = cv2.add(img, laplacian)

    # 显示原图像和锐化后的图像
    cv2.imshow('Original', img)
    cv2.imshow('Sharpened', sharpened)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogramEqualization(img_path):

    # 读取图像
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 全局直方图均衡化
    global_histogram_equalization = cv2.equalizeHist(image)

    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_histogram_equalization = clahe.apply(image)

    # 显示原始图像、全局直方图均衡化后的图像和自适应直方图均衡化后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Global Histogram Equalization', global_histogram_equalization)
    cv2.imshow('Adaptive Histogram Equalization', adaptive_histogram_equalization)

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_noisy_path = "img/noisy_chunk.jpg"
    image_rich_path = "img/rich_women.jpg"
    smoothingfilter(image_noisy_path)
    imageSharpening(image_rich_path)
    histogramEqualization(image_rich_path)
