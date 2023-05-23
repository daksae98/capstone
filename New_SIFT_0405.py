import cv2
import numpy as np
import time


# 이미지 로드
img1 = cv2.imread('dataset/100_0033-빌딩-오후/100_0033_0053.JPG')
img2 = cv2.imread('dataset/100_0033-빌딩-오후/100_0033_0054.JPG')




# 4:3 image resize 
# 각 보간법 노션에 정리해놓은 거 참고하여, 각각 비교하면 좋을 듯? 하지만 눈에 띄게 달라지는점은 없는거 같음ㅎ 여기에 시간 쓸 필요 전혀 없을껄..?
# cv2.INTER_NEAREST,cv2.INTER_LINEAR(default),cv2.INTER_CUBIC,cv2.INTER_LANCZOS4,cv2.INTER_AREA 

# 이미지 resize시 원본이미지의 비율대로 줄여야함 아니면 ex)700,525-> 4:3
img1 = cv2.resize(img1, (700, 525), cv2.INTER_LANCZOS4)  
img2 = cv2.resize(img2, (700, 525), cv2.INTER_LANCZOS4)


# img1g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img2g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# img3 = np.array(img1g)-np.array(img2g);
# print(img3)
# img3 = (img3 - img3.min())/(img3.max() - img3.min())
# cv2.imshow("sub",img3)
# cv2.waitKey()


start_time = time.time() # Time.1시작 시간 저장, 이미지에서 keypoint descriptor 추출시간 측정 start
# SIFT 알고리즘 객체 생성
sift = cv2.xfeatures2d.SIFT_create()

# 키 포인트와 디스크립터 추출

kp1, des1 = sift.detectAndCompute(img1, None) #detectAndCompute-이미지에서 keypoint,descriptor추출하는 함수
kp2, des2 = sift.detectAndCompute(img2, None)

print(f'첫번째 이미지 keypoint:{len(kp1)}개 \n')
print(f'두번째 이미지 keypoint:{len(kp2)}개 \n')

end_time = time.time() # Time.1종료 시간 저장
print(f"Time.1_Keypoint 및 Descriptor 추출에 걸린 시간: {end_time - start_time:.5f}초")

# 키 포인트 그리기 
img_draw1 = cv2.drawKeypoints(img1, kp1, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_draw2 = cv2.drawKeypoints(img2, kp2, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("Image1 Keypoint result.png", img_draw1)
# cv2.namedWindow("Image1 Keypoint result",cv2.WINDOW_NORMAL)
cv2.imshow("Image1 Keypoint result", img_draw1)
# cv2.resizeWindow("Image1 Keypoint result",500,500)
cv2.imwrite("Image2 Keypoint result.png", img_draw2)
# cv2.namedWindow("Image2 Keypoint result",cv2.WINDOW_NORMAL)
cv2.imshow("Image2 Keypoint result", img_draw2)
# cv2.resizeWindow("Image2 Keypoint result",500,500)

cv2.waitKey()
cv2.destroyAllWindows()


start_time = time.time() # Time.2시작 시간 저장 ->매칭 알고리즘

# 매칭 알고리즘 객체 생성 -> BFMatcher 생성, Hamming거리, 상호 체크 Hamming거리 변환에 따른 ? 사실 이것도 미미할듯..
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# 모든 매칭 결과 추출
matches = bf.match(des1, des2) #SIFT의 디스크립터 매칭 결과를 저장하는 리스트

print(f'총 매칭 keypoints matched: {len(matches)}개 ')

end_time = time.time() # Time.2 종료 시간 저장
print(f"Time.2_Feature 매칭에 걸린 시간:{end_time - start_time:.5f}초")

# 모든 매칭 결과 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# 모든 매칭 결과 그림 출력 
cv2.imwrite('Total SIFT matches result.png', res)
cv2.imshow('Total SIFT matches result', res)
cv2.waitKey()
cv2.destroyAllWindows()

# 매칭 결과를 거리 값에 따라 오름차순으로 정렬
#오름차순=거리값이 작을수록(두 디스크립터 유사도가 높다의 의미정도?) 매칭이 잘되었다고 판단할 수 있음.
matches = sorted(matches, key=lambda x: x.distance)



start_time = time.time() #Time.3 시작 시간 저장 ->RANCAC 시작시점 inliersfmf dnlgks
MIN_MATCH_COUNT = 10 #임계값=> 최소 매칭 픽셀 수, 즉 두 이미지 간에 일치하는 최소 매칭 수
#10.0->RANSAC에서 사용될 임계값, 매칭된 특징점들 사이의 거리. 즉,임계값이 크면 inlier로 인식되는 수 많아짐 <-> 작으면 inlier 수 줄어듬
if len(matches) > MIN_MATCH_COUNT:
    
    src_pts = [kp1[match.queryIdx].pt for match in matches] #queryIdx:첫번째 이미지의 특징점 인덱스
    dst_pts = [kp2[match.trainIdx].pt for match in matches] #trainIdx:두번째 이미지의 특징점 인덱스
    M, mask = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, 10.0) 
    matches_mask = mask.ravel().tolist()

    inliers = []
    for i, match in enumerate(matches):
        if matches_mask[i]:
            inliers.append(match)
    
    end_time = time.time() # Time.3 종료 시간 저장
    print(f"Time.3_Inliers 추출에 걸린 시간: {end_time - start_time:.5f}초")
    # 정상점(INLIERS)의 개수 출력
    print(f'Inliers(good matched):{len(inliers)} ')

    # 정상점(INLIERS)만 그리기 위한 이미지 생성
    img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

   
    # 결과 이미지 출력
    cv2.imwrite('INliers SIFT result.png', img_inliers)
    cv2.imshow('INliers SIFT result', img_inliers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  
       
else:
    print("제대로 구현이 안될 경우 - {}/{}".format(len(matches), MIN_MATCH_COUNT))


    