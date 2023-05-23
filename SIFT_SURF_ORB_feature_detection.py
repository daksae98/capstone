#################함수정리##########################

# 1.SIFT =>SIFT는 이미지 피라미드를 이용해서 크기 변화에 따른 특징점 검출 문제를 해결한 알고리즘 /
#        => 크기 변화에 따른 특징 검출 문제를 해결하기 위해 이미지 피라미드를 사용하므로 속도가 느리다는 단점 이에 개선된 것이 SURF
# detector = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
# nfeatures: 검출 최대 특징 수
# nOctaveLayers: 이미지 피라미드에 사용할 계층 수
# contrastThreshold: 필터링할 빈약한 특징 문턱 값
# edgeThreshold: 필터링할 엣지 문턱 값
# sigma: 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값
#===============================================================================================
#===============================================================================================
# 2.SURF =>이미지 피라미드 대신 필터의 크기를 변화시키는 방식으로 성능을 개선한 알고리즘
# ex) 100,3,True,Ture
# detector = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
# hessianThreshold(optional): 특징 추출 경계 값 (default=100)
# nOctaves(optional): 이미지 피라미드 계층 수 (default=3)
# extended(optional): 디스크립터 생성 플래그 (default=False), True: 128개, False: 64개
# upright(optional): 방향 계산 플래그 (default=False), True: 방향 무시, False: 방향 적용
#===============================================================================================
#===============================================================================================
# 3. ORB =>BRIEF에 방향과 회전을 고려하도록 개선한 알고리즘
# BRIEF는 특징점 검출은 지원하지 않는 디스크립터 추출기
# ORB는 특징점 검출 알고리즘으로 FAST를 사용하고 회전과 방향을 고려하도록 개선했으며 속도도 빨라 SIFT와 SURF의 좋은 대안
# ex)
# detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold)
# nfeatures(optional): 검출할 최대 특징 수 (default=500)
# scaleFactor(optional): 이미지 피라미드 비율 (default=1.2)
# nlevels(optional): 이미지 피라미드 계층 수 (default=8)
# edgeThreshold(optional): 검색에서 제외할 테두리 크기, patchSize와 맞출 것 (default=31)
# firstLevel(optional): 최초 이미지 피라미드 계층 단계 (default=0)
# WTA_K(optional): 임의 좌표 생성 수 (default=2)
# scoreType(optional): 특징점 검출에 사용할 방식 (cv2.ORB_HARRIS_SCORE: 해리스 코너 검출(default), cv2.ORB_FAST_SCORE: FAST 코너 검출)
# patchSize(optional): 디스크립터의 패치 크기 (default=31)
# fastThreshold(optional): FAST에 사용할 임계 값 (default=20)


#======================CODE START===================================================
#======================CODE START===================================================

import cv2
import numpy as np

# # 1. SIFT로 특징점 및 디스크립터 추출
# img = cv2.imread('C:\YJpython\sensorsystem\Book1.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # SIFT 추출기 생성
# sift = cv2.xfeatures2d.SIFT_create()
# # 키 포인트 검출과 서술자 계산
# keypoints, descriptor = sift.detectAndCompute(gray, None)
# print('keypoint:',len(keypoints), 'descriptor:', descriptor.shape)
# print(descriptor)

# # 키 포인트 그리기
# img_draw = cv2.drawKeypoints(img, keypoints, None, \
#                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # 결과 출력
# cv2.imshow('SIFT', img_draw)
# cv2.waitKey()
# cv2.destroyAllWindows()

#===============================================================================================
#===============================================================================================

# 2.SURF로 특징점 및 특징 디스크립터 추출 

# import cv2
# import numpy as np

# img = cv2.imread('C:\YJpython\sensorsystem\Book1.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # SURF 추출기 생성 ( 경계:1000, 피라미드:3, 서술자확장:True, 방향적용:True)
# surf = cv2.xfeatures2d.SURF_create(1000, 3, True, True)
# # 키 포인트 검출 및 서술자 계산
# keypoints, desc = surf.detectAndCompute(gray, None)
# print(desc.shape, desc)
# # 키포인트 이미지에 그리기
# img_draw = cv2.drawKeypoints(img, keypoints, None, \
#                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow('SURF', img_draw)
# cv2.waitKey()
# cv2.destroyAllWindows()

#===============================================================================================
#===============================================================================================

# 3.ORB로 특징점 및 특징 디스크립터 검출 

import cv2
import numpy as np

img = cv2.imread('C:\YJpython\sensorsystem\Book1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB 추출기 생성
orb = cv2.ORB_create()
# 키 포인트 검출과 서술자 계산
keypoints, descriptor = orb.detectAndCompute(img, None)
# 키 포인트 그리기
img_draw = cv2.drawKeypoints(img, keypoints, None, \
             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 결과 출력
cv2.imshow('ORB', img_draw)
cv2.waitKey()
cv2.destroyAllWindows()