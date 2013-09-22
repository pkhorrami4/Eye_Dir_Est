import numpy as np
import cv2
import cv2.cv as cv
import copy
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--pt_bnds", dest='pt_vect',  nargs = '*', default=[])
args = parser.parse_args()

def display_pts(img, RAND_FLAG, pt_vect, label_dict):

	if(RAND_FLAG):			
		# Random Pattern
		[col_ind, row_ind] = np.meshgrid( np.linspace(40, N-40, 24).astype('int'), np.linspace(40, M-40, 14).astype('int')  )
		pt_list_orig = np.vstack( (row_ind.flatten(), col_ind.flatten())).T
		#print(pt_list.shape)
		ind = range(0, pt_list_orig.shape[0])
		#print(ind[0:10])
		ind = np.random.permutation(ind)
		#print(ind[0:10])
		pt_list = pt_list_orig[ind[pt_vect],:]
		#print('Original Point List: {}'.format(pt_list_orig[ind,:]))
		#out_file = open('./Train_Labels/orig_pts.pkl', 'wb')
		#pickle.dump(pt_list_orig[ind,:], out_file)
		#out_file.close()
	else:
		# Grid
		[col_ind, row_ind] = np.meshgrid( np.linspace(120, 1800, 4).astype('int'), np.linspace(60,1020, 4).astype('int') )
		pt_list = np.vstack( (row_ind.flatten(), col_ind.flatten())).T
		pt_list = pt_list[pt_vect]
	
	for i in range(0, pt_list.shape[0]):
		pt = pt_list[i,:]
		print('pt: '+str(pt))
		DOT_IMG = copy.deepcopy(img)
		
		# Display blinking dot
		blink_dot(DOT_IMG, (pt[1], pt[0]), pt_vect[i])#, 5)

		
		key = cv.WaitKey(500)		
		#i=i+1
				
		#if(key == 27):
		#	break

		#if(i == pt_list.shape[0]):
			#break;
			
		print('\n')
		
		#label_dict[pt_vect[i]] = pt
		
		
	cv.DestroyWindow("CALIB_SCREEN")

	for i in range(0, pt_list_orig.shape[0]):
		label_dict[ind[i]] = pt_list_orig[i,:]
	
	
def blink_dot(img, pt, i):#, num_blinks):
	print('In Blink Dot')
	
	#for i in range(0,num_blinks):
	while 1:
		cv2.circle(img, pt, 11, cv.Scalar(255, 255, 255), -1)
		cv2.imshow("CALIB_SCREEN", img)
		key = cv.WaitKey(500)
		
		img = np.zeros( img.shape)
		cv2.imshow("CALIB_SCREEN", img)
		key = cv.WaitKey(500)
		
		if(key == 27):
			print('Dot Skipped!!')
			return
			
		if(key == 99):
			# If c is pressed, capture images
			capture_imgs(i)
			
			# Blink a Green Dot
			cv2.circle(img, pt, 11, cv.Scalar(0, 255, 0), -1)
			cv2.imshow("CALIB_SCREEN", img)
			key = cv.WaitKey(500)
			
			img = np.zeros( img.shape)
			cv2.imshow("CALIB_SCREEN", img)
			key = cv.WaitKey(500)
						
			return
	

def capture_imgs(i):
	print('In Capture Images!!')

	num_imgs_cap = 1
	
	READ_FLAG, RGB_img = vid_capt.read()
	for j in range(0, num_imgs_cap):
		READ_FLAG, RGB_img = vid_capt.read()		
		RGB_img = detect_face(RGB_img)		
		cv2.imwrite('./Train_Frames/cap_'+str(i).zfill(3)+'_img'+str(j)+'.png', RGB_img)
	
	
def detect_face(RGB_img):
	# Image Properties
	#print('Image Size: ({},{})'.format(cv.GetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FRAME_HEIGHT), cv.GetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FRAME_WIDTH)))
	#print('FPS: {}'.format(cv.GetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FPS)))

	RGB_img_mat = cv.fromarray(RGB_img)
	
	haarFace = cv.Load('haarcascade_frontalface_default.xml')

	#RGB_img = cv.fromarray(np.array(rgb[:,:,::-1]))
	allFaces = cv.HaarDetectObjects(RGB_img_mat, haarFace, cv.CreateMemStorage(), scale_factor=1.1, min_neighbors=10, flags=0, min_size=(50,50))

	# Get confidences
	if(allFaces != []):

		#print(allFaces)
		face_confid = [c for ((x,y,w,h), c) in allFaces]
		area  = [w*h for ((x,y,w,h), c) in allFaces]
		
		#max_ind = np.argmax(face_confid)
		max_ind = np.argmax(area)
		FINAL_FACE = allFaces[max_ind]


		x0 = FINAL_FACE[0][0]
		y0 = FINAL_FACE[0][1]
		w = FINAL_FACE[0][2]
		h = FINAL_FACE[0][3]

		# Show detected face
		print('Face Detected!!')
		#cv.Rectangle(RGB_img_mat, (x0, y0), (x0+w, y0+h), cv.RGB(0,0,255), 2)
		
		# Detect eyes only in given face region
		print('Face: '+str(FINAL_FACE))
		cropped_img = RGB_img[y0:y0+h, x0:x0+w]
		
		#cv.Smooth(cropped_img, cropped_img, cv.CV_GAUSSIAN, 15, 15)
		#print(cv.GetSize(cropped_img))
		#cv.ShowImage('crop', cropped_img)
		#cv.SaveImage('IMAGE.png', cropped_img)
		#allEyes = detect_eyes(cropped_img)
		
		#print('Eyes: '+str(allEyes))
		#for eye in allEyes:
		#   eye = eye[0]
		#	eye=(eye[0]+x0, eye[1]+y0, eye[2], eye[3])
		#	cv.Rectangle(RGB_img, (eye[0], eye[1]), (eye[0]+eye[2], eye[1]+eye[3]), cv.RGB(255,0,0), 2)
		
		
		return np.asarray(cropped_img[:,:])
		
	else:
		print('No Face!!')		
		return RGB_img


## MAIN

if __name__ == '__main__':


	pt_vect = args.pt_vect
	label_dict = {}
	print(pt_vect)
	g = raw_input('Waiting...')

	if(pt_vect == []):
		pt_vect = 0
	else:
		pt_vect = range(int(pt_vect[0]), int(pt_vect[1]))
	

	np.random.seed(5)
	
	vid_capt = cv2.VideoCapture(0)
	vid_capt.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	vid_capt.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
	vid_capt.set(cv.CV_CAP_PROP_FPS,30)
	
	
	M = 1080
	N = 1920
	SCREEN = np.zeros( (M, N, 3) ) # Black Screen

	cv2.namedWindow("CALIB_SCREEN", cv.CV_WINDOW_NORMAL)
	cv2.moveWindow("CALIB_SCREEN", 0, 0)
	cv2.setWindowProperty("CALIB_SCREEN", cv2.WND_PROP_FULLSCREEN, cv.CV_WINDOW_FULLSCREEN)

	cv2.imshow("CALIB_SCREEN", SCREEN)

	#display_grid(SCREEN)
	#display_rand_pattern(SCREEN)
	RAND_FLAG = 1
	display_pts(SCREEN, RAND_FLAG, pt_vect, label_dict)
	
	
	vid_capt.release()
	
	
	#print(label_dict)
	#out_file = open('./Train_Labels/labels.pkl', 'wb')
	#pickle.dump(label_dict, out_file)
	#out_file.close()