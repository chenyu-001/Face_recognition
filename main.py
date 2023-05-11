import cv2
import matplotlib.pyplot as plt


def filterEnhance(image_path):
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


if __name__ == "__main__":
    image_path = "img/noisy_chunk.jpg"
    filterEnhance(image_path)
