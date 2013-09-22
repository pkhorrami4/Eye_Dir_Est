import numpy as np
from scipy.misc import imresize
import cv2
import cv2.cv as cv
import os


def get_order(eye0, eye1):

	if(eye0[0][0] < eye1[0][0]):
		right_eye = eye0
		left_eye = eye1
	else:
		right_eye = eye1
		left_eye = eye0		
		
	return right_eye, left_eye



if __name__ == '__main__':

	face_file_list = os.listdir('./Train_Frames/')
	#print(face_file_list)
	cv2.namedWindow("FACE", 1)
	cv2.moveWindow("FACE", 20, 500)
	cv2.namedWindow("RIGHT", 1)
	cv2.moveWindow("RIGHT", 300, 500)
	cv2.namedWindow("LEFT", 1)
	cv2.moveWindow("LEFT", 500, 500)
	cv2.namedWindow("BOTH", 1)
	cv2.moveWindow("BOTH", 700, 500)
		
	haarEyes = cv.Load('haarcascade_eye.xml')
	storage = cv.CreateMemStorage()

	for file in face_file_list:
		print(file)
		
		filename_parts = file.split('_')
		img_num = filename_parts[1]
		print(img_num)
		
		F_arr = cv2.imread('./Train_Frames/'+file)
		F = cv.fromarray(F_arr)
		
		#haarEyes = cv2.CascadeClassifier('haarcascade_eye.xml')
		#detectedEyes = haarEyes.detectMultiScale(F, scaleFactor=1.1, minNeighbors=10, flags=0, minSize=(20,20), maxSize=(50,50))
		detectedEyes = cv.HaarDetectObjects(F, haarEyes, storage, scale_factor=1.1, min_neighbors=15, flags=0, min_size=(20,20))
		#print(detectedEyes)
		
		
		print(str(len(detectedEyes))+' eyes detected')
		
		eye_confid = [c for ((x,y,w,h), c) in detectedEyes]
		eye_confid_inds = np.argsort(eye_confid)[::-1][:2]
			
		#print(eye_confid_inds)
		eye0 = detectedEyes[eye_confid_inds[0]]
		eye1 = detectedEyes[eye_confid_inds[1]]                            
		
		[right_eye, left_eye] = get_order(eye0, eye1)
		
		# Display bounding boxes
		#cv2.rectangle(F_arr, (right_eye[0][0], right_eye[0][1]), (right_eye[0][0]+right_eye[0][2], right_eye[0][1]+right_eye[0][3]), cv.Scalar(0,255,0), 2)
		#cv2.rectangle(F_arr, (left_eye[0][0], left_eye[0][1]), (left_eye[0][0]+left_eye[0][2], left_eye[0][1]+left_eye[0][3]), cv.Scalar(0,0,255), 2)
		
		
		print('Right Eye: '+str(right_eye))
		print('Left Eye: '+str(left_eye))
		
		#right_eye_img = []
		#left_eye_img = []
		
		right_eye_img = F_arr[ right_eye[0][1]-5:right_eye[0][1]+right_eye[0][3]+5, right_eye[0][0]:right_eye[0][0]+right_eye[0][2]+10 ].copy()
		left_eye_img  = F_arr[ left_eye[0][1]-5:left_eye[0][1]+left_eye[0][3]+5, left_eye[0][0]-10:left_eye[0][0]+left_eye[0][2] ].copy()

		right_eye_img = imresize(right_eye_img, (100,100,3), interp='bicubic')
		left_eye_img = imresize(left_eye_img, (100,100,3), interp='bicubic')
		both_eye_img = np.hstack( (right_eye_img, left_eye_img) )
		
		#print(right_eye_img.shape)
		#print(left_eye_img.shape)
		
		#cv2.imshow("FACE", F_arr)		
		#cv2.imshow("RIGHT", right_eye_img)
		#cv2.imshow("LEFT", left_eye_img)
		#cv2.imshow("BOTH", both_eye_img)
		
		#key = cv2.waitKey(0)
		
		#if(key == 27):	
		#	break

		#cv2.imwrite('./right_eye.png', np.asarray(right_eye_img[:,:]))
		#cv2.imwrite('./left_eye.png', np.asarray(left_eye_img[:,:]))
		cv2.imwrite('./Train_Eyes/eyes_'+img_num+'.png', both_eye_img)

	cv2.destroyAllWindows()		