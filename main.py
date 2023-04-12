import cv2


def create_and_show_mirror_image(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 检查图片是否正确读取
    if image is None:
        print("Error: Cannot read the image. Please check the file path.")
        return

    # 创建镜像图片
    mirror_image = cv2.flip(image, 1)

    # 显示原始图片
    cv2.imshow('Original Image', image)

    # 显示镜像图片
    cv2.imshow('Mirror Image', mirror_image)

    # 等待按键，然后关闭显示窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "img/rich_women.jpg"
    create_and_show_mirror_image(image_path)
