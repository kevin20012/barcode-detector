import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import find_corner, alpha_trim_filter, detect_barcode


if __name__ == '__main__':
    img_color = cv2.imread('barcode1.jpg')
    img_color = cv2.resize(img_color, (750, 550))
    input_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    

    img1, img2, img3, img4, img5, img6 = detect_barcode(img_color, input_img, './result_barcode1.jpg')

    plt.subplot(161)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('GaussianBlur')
    plt.subplot(162)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Sobel')
    plt.subplot(163)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Thresholding')
    plt.subplot(164)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Median filter')
    plt.subplot(165)
    plt.imshow(img5, cmap='gray')
    plt.axis('off')
    plt.title('Dilate 2 times and Find Barcode candidate')
    plt.subplot(166)
    plt.imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Result')
    plt.tight_layout()
    plt.show()
