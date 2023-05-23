import cv2 
import numpy as np
import sys

# 9.880273708015928
# dataset/100_0030-운동장-정오/100_0030_0002.JPG

# 6.74536110269506
src = cv2.imread('/Users/hyunsukim/Desktop/23-1/종설/code/dataset/100_0031-빌딩-정오/100_0031_0014.JPG', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('image load failed!')
    sys.exit()
    
# 필터 마스크 생성
kernel = np.ones((3, 3), dtype=np.int8) # 9.이 실수이므로 자동으로 float로 설정되지만 데이터 타입을 지정해주었습니다.
kernel[0][0] = 0
kernel[0][2] = 0
kernel[1][1] = -4
kernel[2][0] = 0
kernel[2][2] = 0
print(kernel)


rows,columns = src.shape

resized_src = cv2.resize(src,(rows//3,columns//3),interpolation=cv2.INTER_AREA)

print(resized_src.shape)
dst = cv2.filter2D(resized_src, -1, kernel) # -1은 입력 영상과 동일한 데이터의 출력 영상 생성
# np.histogram(dst,bins=10)
print(kernel)

# mean_kernel = np.ones((rows//30,columns//30),dtype=np.float16)/9
# print(mean_kernel.shape)
# mean_d = cv2.filter2D(dst,-1,mean_kernel)

# sys.exit()
cv2.imshow('src', resized_src)
cv2.waitKey()
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.imshow('add',resized_src + dst)
cv2.waitKey()
print(np.mean(abs(dst)))
# cv2.imshow('mean_d', mean_d)
# cv2.waitKey()