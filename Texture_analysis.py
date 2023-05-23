import cv2
import numpy as np
import skimage.feature as feature

'''
result = {
    image_name : 'asdf.jpg'
    texture : GLCM result
    SIFT_keypoint : num
    KAZE_keypoint : num
    ORB_keypoint : num => only 500..

}
아 아니면 매칭 별로 두 영상의 texture의 평균? 어차피 연결되어있으니까..

'''

image_file = '/Users/hyunsukim/Desktop/23-1/종설/code/dataset/100_0030-운동장-정오/100_0030_0003.JPG'
image_spot = cv2.imread(image_file)
gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)

# Find the GLCM


# Param:
# source image
# List of pixel pair distance offsets - here 1 in each direction
# List of pixel pair angles in radians
graycom = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

# Find the GLCM properties
contrast = feature.graycoprops(graycom, 'contrast')
dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
homogeneity = feature.graycoprops(graycom, 'homogeneity')
energy = feature.graycoprops(graycom, 'energy')
correlation = feature.graycoprops(graycom, 'correlation')
ASM = feature.graycoprops(graycom, 'ASM')

print("Contrast: {}".format(np.mean(contrast)))
print("Dissimilarity: {}".format(np.mean(dissimilarity)))
print("Homogeneity: {}".format(np.mean(homogeneity)))
print("Energy: {}".format(np.mean(energy)))
print("Correlation: {}".format(np.mean(correlation)))
print("ASM: {}".format(np.mean(ASM)))