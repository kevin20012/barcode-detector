# salt-and-pepper noise reduction!
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import alpha_trim_filter, find_corner, detect_barcode



if __name__ == '__main__':
    img_color = cv2.imread('barcode2.jpg')
    img_color = cv2.resize(img_color, (750, 550))
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    """
    Salt&pepper noise 제거
    """
    filter = alpha_trim_filter(kernel_size=9, mode='median', th=0.8)
    img_1 = filter.filter(img_gray)

    """
    Find barcode
    """
    img_2, img3, img4, img5, img6, img7 = detect_barcode(img_color, img_1, './result_barcode2.jpg')
    
    plt.subplot(131)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img_1, cmap='gray')
    plt.title('Remove salt&pepper noise')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img_2, cmap='gray')
    plt.title('Thresholding')
    plt.axis('off')
    plt.show()

    plt.subplot(161)
    plt.imshow(img_2, cmap='gray')
    plt.axis('off')
    plt.title('Image removed noises')
    plt.subplot(162)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Sobel')
    plt.subplot(163)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Thresholding')
    plt.subplot(164)
    plt.imshow(img5, cmap='gray')
    plt.axis('off')
    plt.title('Median filter')
    plt.subplot(165)
    plt.imshow(img6, cmap='gray')
    plt.axis('off')
    plt.title('Dilate 2 times and Find Barcode candidate')
    plt.subplot(166)
    plt.imshow(cv2.cvtColor(img7, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Result')
    plt.tight_layout()
    plt.show()