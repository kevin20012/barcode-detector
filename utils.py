import numpy as np
import cv2
############################################ Freq domain utils ############################################

# 주파수 도메인 utils
class Freq_utils():
    def zero_padding(self, img):
        # Zero-padding the image
        M, N = img.shape

        P, Q = 2 * M, 2 * N  # New dimensions for zero-padding
        padded_image = np.zeros((P, Q), dtype=np.uint8)
        padded_image[0:M, 0:N] = img
        return padded_image

    def centering(self, img):
        P, Q = img.shape
        padded_image_new = np.zeros((P, Q))
        for x in range(P):
            for y in range(Q):
                padded_image_new[x, y] = img[x, y] * ((-1)**(x+y))
        return padded_image_new

    def manual_filter(self, image):
        M, N = image.shape
        H = np.ones((M,N))
        
        interval = 50
        move = 200
        
        for u in range(M):
            for v in range(N):
                if u==v:
                    start, end = max(move+v-interval//2, 0), min(move+v+interval//2, N-1)
                    H[u, start:end] = 0

        for u in range(M):
            for v in range(N):
                if M-u==v:
                    start, end = max(move+v-interval//2, 0), min(move+v+interval//2, N-1)
                    H[u, start:end] = 0

        # radius = 200
        # center = (M//2, N//2)
        # for u in range(M):
        #     for v in range(N):
        #         if (u-center[0])**2+(v-center[1])**2 <= radius**2:
        #             H[u,v] = 0

        # for u in range(M):
        #     for v in range(N):
        #         if v > center[1]-30 and v < center[1]+30:
        #             H[u,v]=0


        return H
    
############################################ Alpha trimmed mean filter ############################################

# Alpha trimmed mean filter
class alpha_trim_filter():
    def __init__(self, kernel_size=9, mode='arithmatic', th=0.8):
        self.kernel_size = kernel_size
        self.th = th
        self.mode = mode

    def edge_padding(self, img):
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd number!")

        new_pad = self.kernel_size//2
        img = np.pad(img, ((new_pad, new_pad), (new_pad, new_pad)), mode='edge')
        return img

    def filter(self, img):
        img_padded = self.edge_padding(img)

        result_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i in range(img_padded.shape[0]-self.kernel_size+1):
            for j in range(img_padded.shape[1]-self.kernel_size+1):
                flat_value = np.sort(img_padded[i:i+self.kernel_size, j:j+self.kernel_size].flatten())

                start, end = int((self.kernel_size*self.kernel_size)*(self.th/2)), int((self.kernel_size*self.kernel_size)*(1-self.th/2))
                # print(start, end)
                if self.mode=='arithmatic':
                    slice_values_and_mean = flat_value[start:end].mean()
                elif self.mode=='geometric':
                    log_data = np.log(flat_value + 1e-10)
                    slice_values_and_mean = np.exp(np.mean(log_data))
                elif self.mode=='median':
                    slice_values_and_mean = flat_value[len(flat_value)//2]
                # print(slice_values_and_mean, flat_value[start:end])

                result_img[i, j] = slice_values_and_mean
        return result_img
    
############################################ Utils ############################################

# 이진 이미지로부터 왼쪽상단 pt1, 오른쪽하단 pt2 좌표 검출기
def find_corner(img):
    """
    원점과 가장 가까운 점 -> pt1
    오른쪽 맨 하단과 가까운 점 -> pt2
    return
        pt1 : 왼쪽상단좌표
        pt2 : 오른쪽하단좌표
    """
    M, N = img.shape[0], img.shape[1]

    dist1 = np.sqrt(M**2+N**2)
    dist2 = dist1
    pt1 = (0,0)
    pt2 = (0,0)

    for i in range(M):
        for j in range(N):
            if img[i,j] > 0:
                cur_dist1 = np.sqrt(i**2+j**2)
                cur_dist2 = np.sqrt((i-M)**2+(j-N)**2)

                if cur_dist1 < dist1:
                    pt1 = (j,i)
                    dist1 = cur_dist1
                
                if cur_dist2 < dist2:
                    pt2 = (j,i)
                    dist2 = cur_dist2

            
    return pt1, pt2

def ContrastEnhancement(img):
    
    # intencity: 기울기, bias: y절편
    intensity, bias = 1.41, -28.2

    # Manual Contrast Enhancement 구현
    H, W = img.shape
    result = np.zeros((H,W))

    for i in range(H):
        for j in range(W):
            if img[i,j] < 20:
                result[i,j] = 0
            elif img[i,j] < 200:
                result[i,j] = img[i,j]*intensity+bias
            else:
                result[i,j] = 230

    result = result.astype(np.uint8)

    return result 

def pick_barcode_candi(img, kernel=(10, 20), th=0.7):
    M, N = kernel[0], kernel[1]
    result_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for x in range(img.shape[0]-M):
        for y in range(img.shape[1]-N):
            binary = np.where(img[x:x+M,y:y+N]>100, 1, 0)
            count_ratio = binary.mean()
            # print(count_ratio)
            if count_ratio > th:
                result_img[x:x+M,y:y+N] = 255
    return result_img


def detect_barcode(img_color, input_img, save_path):
    # 노이즈 제거
    print('✅ 노이즈 제거')
    img1 = cv2.GaussianBlur(input_img, ksize=(5,5), sigmaX=0)
    # 수직방향 엣지 검출 x 방향 1차 미분 이용 sobel 연산
    print('✅ Sobel 연산 - 수직방향 엣지 강조')
    img2 = cv2.Sobel(img1, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    img2 = np.absolute(img2)
    img2 = np.uint8(255 * img2 / np.max(img2))

    # Thresholding
    print('✅ Otsu\'s Thresholding')
    ret, img3 = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)
    

    # 자잘한 점 노이즈들 제거
    print('✅ Remove salt&pepper noise')
    filter = alpha_trim_filter(kernel_size=5, mode='median', th=0.8)
    img4 = filter.filter(img3)

    # Dilation으로 세그멘테이션
    print('✅ Dilation 및 바운딩 박스 후보 찾기')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img5 = cv2.morphologyEx(img4, cv2.MORPH_DILATE, kernel, iterations=2)
    img5 = pick_barcode_candi(img5, kernel=(50,100), th=0.90)


    # 바운딩박스를 위한 코너 검출
    print('✅ Find bounding box')
    
    margin = int(input_img.shape[0]*0.05)
    pt1, pt2 = find_corner(img5)
    print(pt1, pt2)
    pt1 = (pt1[0]-margin, pt1[1]-margin)
    pt2 = (pt2[0]+margin, pt2[1]+margin)
    img6 = cv2.rectangle(img_color, pt1, pt2, color=(0,255,0), thickness=10)
    cv2.imwrite(save_path, img6)

    return img1, img2, img3, img4, img5, img6