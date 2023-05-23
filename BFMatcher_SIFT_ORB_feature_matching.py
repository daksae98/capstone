import cv2, numpy as np
#===============================================================================================
#===============================================================================================
#함수정리.1 => 1. detector.compute / 2.detector.detectAndCompute 정리
#
# keypoints, descriptors = detector.compute(image, keypoins, descriptors)
# => 특징점을 전달하면 특징 디스크립터를 계산해서 반환
# keypoints, descriptors = detector.detectAndCompute(image, mask, decriptors, useProvidedKeypoints)
# => 특징점 검출과 특징 디스크립터 계산을 한 번에 수행 -> compute보다 편리
# image: 입력 이미지
# keypoints: 디스크립터 계산을 위해 사용할 특징점
# descriptors(optional): 계산된 디스크립터
# mask(optional): 특징점 검출에 사용할 마스크
# useProvidedKeypoints(optional): True인 경우 특징점 검출을 수행하지 않음
#===============================================================================================
#===============================================================================================
# 함수정리.2 => 1. matcher.match 2. matcher.knnMatch 3. matcher.radiusMatch

# 1. matcher.match(queryDescriptors, trainDescriptors, mask):** 1개의 최적 매칭
# queryDescriptors: 특징 디스크립터 배열, 매칭의 기준이 될 디스크립터
# trainDescriptors: 특징 디스크립터 배열, 매칭의 대상이 될 디스크립터
# mask(optional): 매칭 진행 여부 마스크
# matches: 매칭 결과, DMatch 객체의 리스트
# 2. matcher.knnMatch(queryDescriptors, trainDescriptors, k, mask, compactResult):** k개의 가장 근접한 매칭
# k: 매칭할 근접 이웃 개수
# compactResult(optional): True: 매칭이 없는 경우 매칭 결과에 불포함 (default=False)
# 3. matcher.radiusMatch(queryDescriptors, trainDescriptors, maxDistance, mask, compactResult):**maxDistance 이내의 거리 매칭
# maxDistance: 매칭 대상 거리
#===============================================================================================
#===============================================================================================
# 함수정리.3 => drawMatches => 매칭 결과를 시각적으로 표현하기 위해 두 이미지를 하나로 합쳐서, 매칭점끼리 선으로 연결하는 작업
# cv2.drawMatches(img1, kp1, img2, kp2, matches, flags): 매칭점을 이미지에 표시
# img1, kp1: queryDescriptor의 이미지와 특징점
# img2, kp2: trainDescriptor의 이미지와 특징점
# matches: 매칭 결과
# flags: 매칭점 그리기 옵션 (cv2.DRAW_MATCHES_FLAGS_DEFAULT: 결과 이미지 새로 생성(default값), 
#                    cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: 결과 이미지 새로 생성 안 함, 
#                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 특징점 크기와 방향도 그리기, 
#                    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: 한쪽만 있는 매칭 결과 그리기 제외)
#===============================================================================================
#===============================================================================================
# 함수정리.4 =>BFMatcher(Brute-Force Matcher) 
# =>Brute-Force 매칭기는 queryDescriptors와 trainDescriptors를 하나하나 확인해 매칭되는지 판단하는 알고리즘.
#    OpenCV에서는 cv2.BFMatcher 클래스로 제공
# matcher = cv2.BFMatcher_create(normType, crossCheck)
# normType: 거리 측정 알고리즘 (cv2.NORM_L1, cv2.NORM_L2(default), cv2.NORM_L2SQR, cv2.NORM_HAMMING, cv2.NORM_HAMMING2)
# crosscheck: 상호 매칭이 되는 것만 반영 (default=False)
# crosscheck가 True 이면 양쪽 디스크립터 모두에게서 매칭이 완성된 것만 반영하므로 불필요한 매칭을 줄일 수 있지만 그만큼 속도가 느려진다.
# SIFT,SURF => NORM_L1, NORM_L2 적합 //// ORB => NORM_HAMMING이 적합

#======================CODE START===================================================
#======================CODE START===================================================


## 1.BFMatcher와 SIFT로 매칭 
# img1 = cv2.imread('C:\YJpython\sensorsystem\Book1.jpg')
# img2 = cv2.imread('C:\YJpython\sensorsystem\Book2.jpg')

# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # SIFT 서술자 추출기 생성 ---①
# detector = cv2.xfeatures2d.SIFT_create()
# # 각 영상에 대해 키 포인트와 서술자 추출 ---②
# kp1, desc1 = detector.detectAndCompute(gray1, None)
# kp2, desc2 = detector.detectAndCompute(gray2, None)

# # BFMatcher 생성, L1 거리, 상호 체크 ---③
# matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# # 매칭 계산 ---④
# matches = matcher.match(desc1, desc2)
# # 매칭 결과 그리기 ---⑤
# res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
#                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# # 결과 출력 
# cv2.imshow('BFMatcher + SIFT', res)
# cv2.waitKey()
# cv2.destroyAllWindows()

#===============================================================================================
#===============================================================================================

## 2.BFMatcher와 SURF로 매칭 

# import cv2
# import numpy as np

# img1 = cv2.imread('C:\YJpython\sensorsystem\Book1.jpg')
# img2 = cv2.imread('C:\YJpython\sensorsystem\Book2.jpg')
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # SURF 서술자 추출기 생성 ---①
# detector = cv2.xfeatures2d.SURF_create()
# kp1, desc1 = detector.detectAndCompute(gray1, None)
# kp2, desc2 = detector.detectAndCompute(gray2, None)

# # BFMatcher 생성, L2 거리, 상호 체크 ---③
# matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# # 매칭 계산 ---④
# matches = matcher.match(desc1, desc2)
# # 매칭 결과 그리기 ---⑤
# res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
#                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# cv2.imshow('BF + SURF', res)
# cv2.waitKey()
# cv2.destroyAllWindows()

#===============================================================================================
#===============================================================================================

# 3.BFMatcher와 ORB로 매칭 

import cv2, numpy as np

img1 = cv2.imread('C:\YJpython\sensorsystem\Book1.jpg')
img2 = cv2.imread('C:\YJpython\sensorsystem\Book2.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB 서술자 추출기 생성 ---①
detector = cv2.ORB_create()
# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# BFMatcher 생성, Hamming 거리, 상호 체크 ---③
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 결과 그리기 ---⑤
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('BFMatcher + ORB', res)
cv2.waitKey()
cv2.destroyAllWindows()