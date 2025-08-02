# Find the problem in freq domain!
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import Freq_utils, ContrastEnhancement, alpha_trim_filter, find_corner, detect_barcode


if __name__ == '__main__':
    img_color = cv2.imread('barcode3.jpg')
    img_color = cv2.resize(img_color, (750, 550))
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    """
    Filtering in Freq. domain
    """
    utils = Freq_utils()
    # zero-padding
    padded_img = utils.zero_padding(img)
    # centering
    centered_img = utils.centering(padded_img)
    # DFT
    dft2d = np.fft.fft2(centered_img)
    dft2d_log = np.log(np.abs(dft2d))
    # Filtering
    H = utils.manual_filter(dft2d)
    G = np.multiply(dft2d, H)
    G_log = np.log(np.abs(G))
    # IDFT
    idft2d = np.fft.ifft2(G)
    # Decentering
    decentered_idft2d = utils.centering(idft2d.real)
    # remove padding
    M, N = decentered_idft2d.shape[0]//2, decentered_idft2d.shape[1]//2
    result = decentered_idft2d[0:M, 0:N]
    result = (result - result.min())/(result.max()-result.min())
    result = result*255

    """
    histogram matching for more brightness
    """
    hist1, _ = np.histogram(result.ravel(), bins=256, range=(0,256))
    # manual contrast enhancement
    result_ = ContrastEnhancement(result)
    # gamma correction
    gamma = 0.5
    result_ = cv2.pow(result_ / 255.0, gamma) * 255
    result_ = result_.astype('uint8')
    hist2, _ = np.histogram(result_.ravel(), bins=256, range=(0,256))
    

    """
    Find barcode
    """
    _, img2, img3, img4, img5, img6 = detect_barcode(img_color, result_, './result_barcode3.jpg')

    plt.subplot(141)
    plt.imshow(dft2d_log.real, cmap='gray')
    plt.title('In Freq. domain')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(H, cmap='gray')
    plt.title('Manual filter')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(G_log, cmap='gray')
    plt.title('G * H')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(result, cmap='gray')
    plt.title('Result')
    plt.axis('off')
    plt.show()

    
    plt.subplot(141)
    plt.hist(hist1, 256, [0,256]) # 어두운 영역에 대부분 몰려있음. 이 경우, 히스토그램 이퀄라이제이션으로 해결이 안됨. 수동으로 맞춰주도록 하겠음.
    plt.title('Before apply Contrast Enhancer hist.')
    plt.subplot(142)
    plt.imshow(result, cmap='gray')
    plt.title('Before apply Contrast Enhancer')
    plt.axis('off')
    plt.subplot(143)
    plt.hist(hist2, 256, [0,256]) # 어두운 영역에 대부분 몰려있음. 이 경우, 히스토그램 이퀄라이제이션으로 해결이 안됨. 수동으로 맞춰주도록 하겠음.
    plt.title('After apply Contrast Enhancer hist.')
    plt.subplot(144)
    plt.imshow(result_, cmap='gray')
    plt.title('After apply Contrast Enhancer')
    plt.axis('off')
    plt.show()

    plt.subplot(161)
    plt.imshow(result_, cmap='gray')
    plt.axis('off')
    plt.title('Image removed noises + contrast enhanced')
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
    plt.title('Median filter 20 times')
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