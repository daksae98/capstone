import cv2
import numpy as np
import time
import os
import csv
import pickle
########################## SETTINGS ###########################
# PATH should be ONLY ENG and ends with "/"
# Image folder path
IMG_PATH = 'dataset/100_0033-빌딩-오후/'
RESIZE = (700, 525)
# SIFT, KAZE, ORB,
ALGO_TYPE = "ORB"
RANSAC = 100
################################################################


ALGO = {
    "SIFT" : cv2.xfeatures2d.SIFT_create,
    "KAZE" : cv2.KAZE_create,
    "ORB" : cv2.ORB_create
}

LOSS_DISTANCE = {
    "SIFT" : cv2.NORM_L1,
    "ORB" : cv2.NORM_HAMMING,
    "KAZE" : cv2.NORM_L2
}

# 영상별 texture 추출

def get_key_points(imageList:list):
    key_points = []
    total_time = 0
    list_length = len(imageList)
    fe_algo = ALGO[ALGO_TYPE]()

    for index, image_path in enumerate(imageList):
        print(f'\r processing: {index+1}/{list_length}',end='')
        image = cv2.imread(IMG_PATH + image_path)
        image = cv2.resize(image, RESIZE, cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

        start_time = time.time()
        kp, des = fe_algo.detectAndCompute(image, None)

        end_time = time.time()
        time_spend = end_time - start_time
        total_time = total_time + time_spend

        


        key_point = {
            'imageName' : image_path,
            'keyPoint':kp,
            'descriptor':des,
            
        }

        key_points.append(key_point)
    return key_points, total_time

def run_matches(keypoints:list, match_order=None)->list:
    match_list = []
    total_time = 0
    keypoints_length = len(keypoints)
    if match_order:
        #  with open(f'{PKL_PATH+algo}.pkl', 'rb') as f:
        #     pkl = pickle.load(f)
        pass
    else:
        for index,keypoint in enumerate(keypoints):
            print(f'\rprocessing:{index+1}/{keypoints_length}',end='')
            if(index < keypoints_length -1 ):
                next_keypoint = keypoints[index+1]

                imgName1 = keypoint["imageName"]
                kp1 = keypoint['keyPoint']
                des1 = keypoint['descriptor']
                
                
                imgName2 = next_keypoint["imageName"]
                kp2 = next_keypoint['keyPoint']
                des2 = next_keypoint['descriptor']
               

                bf = cv2.BFMatcher(LOSS_DISTANCE[ALGO_TYPE], crossCheck=True)
                
                start_time = time.time() 
                                               
                matches = bf.match(des1, des2)

                end_time = time.time() 

                time_spend = end_time - start_time
                total_time = total_time + time_spend

                matchInfo = {
                    "img1":imgName1,
                    "kp1":kp1,
                    "kp1_count":len(kp1),
                    "img2":imgName2,
                    "kp2":kp2,
                    "kp2_count":len(kp2),
                    "matches_count":len(matches),
                    "matches":matches,
                }
                # print(f"#{index+1} match_count :{len(matches)}")
                match_list.append(matchInfo)

    return match_list, total_time


def get_inlier_matches(matchList):

    matchList_inliers = []
    tt = 0
    MIN_MATCH_COUNT = 10 
    length=len(matchList)
    for index,matchInfo in enumerate(matchList):
        print(f'\r{index+1}/{length}', end='')
        kp1 = matchInfo["kp1"]
        kp2 = matchInfo["kp2"]
        matches = matchInfo["matches"]
        
        if len(matches) > MIN_MATCH_COUNT:
            start_time = time.time() 
            src_pts = [kp1[match.queryIdx].pt for match in matches] #queryIdx:첫번째 이미지의 특징점 인덱스
            dst_pts = [kp2[match.trainIdx].pt for match in matches] #trainIdx:두번째 이미지의 특징점 인덱스
            M, mask = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, RANSAC) 
            matches_mask = mask.ravel().tolist()

            inliers = []
            for i, match in enumerate(matches):
                if matches_mask[i]:
                    inliers.append(match)
            
            end_time = time.time() # Time.3 종료 시간 저장
            timespend = end_time - start_time
            tt = tt + timespend
            
            # 정상점(INLIERS)의 개수 출력
            
            matchInfo["inliers"] = inliers
            matchInfo["inliers_count"] = len(inliers)
            matchList_inliers.append(matchInfo)

    return matchList_inliers,tt

  
if __name__ == '__main__':
    
    DIR_LIST = os.listdir(IMG_PATH)
    IMAGE_LIST = []
    
    for dirlist in DIR_LIST:
        ext = os.path.splitext(dirlist)[-1]
        if ext == ".JPG":
            IMAGE_LIST.append(dirlist)
    
    IMAGE_LIST.sort()

    print("\n###################### KEYPOINT ######################\n")
    keypointList,total_time_keypoints = get_key_points(IMAGE_LIST)
    print("\ntotal_time_keypoints:",total_time_keypoints,"s")

    print("\n###################### MATCH ######################\n")
    matchList,total_time_match = run_matches(keypointList)
    print("\ntotal_time_match:",total_time_match,"s")

    print("\n###################### INLIER ######################\n")
    matchList_inliers,tt = get_inlier_matches(matchList)
    # print(matchList_inliers)
    print("\ntotal_time_inlier:",tt,"s\n\n")

    print("\n###################### MATCH_INFO ######################\n")
    # out
    # 평균 Keypoint 개수 : keypoint_count_sum / NUM_MATCHES
    # 평균 tiepoint 개수 : matches_count_sum / NUM_MATCHES
    # 평균 Inliner match 개수 : inliers_count_sum / NUM_MATCHES
    
    # 평균 영상 정합률 : matches_per_keypoint_list
    # 평균 정상점 정합률 : inliers_per_matches_list
   

    # 매칭 횟수
    NUM_MATCHES = len(matchList_inliers)

    keypoint_count_sum = 0
    matches_count_sum = 0
    inliers_count_sum = 0
    matches_per_keypoint_list = []
    inliers_per_matches_list = []
    for matchInfo in matchList_inliers:
        """
        matchInfo = {
                    "img1":imgName1,
                    "kp1":kp1,
                    "kp1_count":len(kp1),
                    "img2":imgName2,
                    "kp2":kp2,
                    "kp2_count":len(kp2),
                    "matches_count":len(matches),
                    "matches":matches,
                    "inliers": inliers,
                    "inliers_count" : len(inliers),
                }
        """

        imgName1 = matchInfo["img1"]
        imgName2 = matchInfo["img2"]
        keypoint_count = matchInfo["kp1_count"] if matchInfo["kp1_count"] >= matchInfo["kp2_count"] else matchInfo["kp2_count"]
        matches_count = matchInfo["matches_count"]
        inliers_count = matchInfo["inliers_count"]

        # 평균 Keypoint 개수 : keypoint_count_sum / NUM_MATCHES
        keypoint_count_sum = keypoint_count_sum + keypoint_count
        # 평균 tiepoint 개수 : matches_count_sum / NUM_MATCHES
        matches_count_sum = matches_count_sum + matches_count
        # 평균 Inliner match 개수 : inliers_count_sum / NUM_MATCHES
        inliers_count_sum = inliers_count_sum + inliers_count

        
        matches_per_keypoint = matches_count/keypoint_count
        inliers_per_matches = inliers_count/matches_count

        matches_per_keypoint_list.append(matches_per_keypoint)
        inliers_per_matches_list.append(inliers_per_matches)


        # print(f"{imgName1} & {imgName2}\nmatches_count: {matches_count}\ninliers_count: {inliers_count}\n영상 정합률:{matches_per_keypoint}\n정상점 정합률:{inliers_per_matches}\n\n")

    matches_per_keypoint_list = np.array(matches_per_keypoint_list)
    inliers_per_matches_list = np.array(inliers_per_matches_list)

    # 평균 Keypoint 개수 : keypoint_count_sum / NUM_MATCHES
    print("평균 Keypoint 수 : ", keypoint_count_sum/NUM_MATCHES )
    # 평균 tiepoint 개수 : matches_count_sum / NUM_MATCHES
    print("평균 tiepoint 수 : ", matches_count_sum/NUM_MATCHES)
    # 평균 Inliner match 개수 : inliers_count_sum / NUM_MATCHES
    print("평균 Inliner match 수: ", inliers_count_sum/NUM_MATCHES)
    print("평균 영상 정합률:",matches_per_keypoint_list.mean()," 최대 : ",matches_per_keypoint_list.max())
    print("평균 정상점 정합률:",inliers_per_matches_list.mean(),' 최대 : ',inliers_per_matches_list.max())

    name = IMG_PATH.split('/')[1]
    f = open(f'csvs/{name}.csv','a', newline='')
    wr = csv.writer(f)
    wr.writerow([f'{ALGO_TYPE}_시간', '평균 정상점 정합률', '최대 정상점 정합률','평균 Inlier match 수','평균 영상 정합률','평균 tiepoint 수','평균 keypoint 수'])
    wr.writerow([total_time_keypoints+total_time_match+tt,inliers_per_matches_list.mean(),inliers_per_matches_list.max(),inliers_count_sum/NUM_MATCHES,matches_per_keypoint_list.mean(),matches_count_sum/NUM_MATCHES,keypoint_count_sum/NUM_MATCHES ])
    f.close()