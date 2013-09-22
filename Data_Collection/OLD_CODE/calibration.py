import numpy as np
import cv2
import cv2.cv as cv
import copy


def capture_image():
	print('In Capture Image')

def display_grid(img):
	# Make dot grid
	#[col_ind, row_ind] = np.meshgrid(range(0,N+320,320), range(0,M+180,180))
	[col_ind, row_ind] = np.meshgrid( np.linspace(120, 1800, 4).astype('int'), np.linspace(60,1020, 4).astype('int') )
	pt_list = np.vstack( (row_ind.flatten(), col_ind.flatten())).T


	i = 0
	while 1:
		pt = pt_list[i,:]
		print('pt: '+str(pt))
		DOT_IMG = copy.deepcopy(img)
		
		# Display blink dot
		ret_flag = blink_dot_grid(DOT_IMG, (pt[1], pt[0]), 5)

		key = cv.WaitKey(500)
		i=i+1
		
		if(ret_flag == 1):
			break
		
		if(key == 27):
			break

		if(i == pt_list.shape[0]):
			break;
			
	cv.DestroyWindow("CALIB_SCREEN")	

def display_rand_pattern(img):

	# Get dot pattern
	[col_ind, row_ind] = np.meshgrid( np.linspace(0, N, 100).astype('int'), np.linspace(0, M, 100).astype('int')  )
	pt_list = np.vstack( (row_ind.flatten(), col_ind.flatten())).T
    #print(pt_list.shape)
	ind = range(0, pt_list.shape[0])
	#print(ind[0:10])
	ind = np.random.permutation(ind)[0:2]
	#print(ind[0:10])
	pt_list = pt_list[ind,:]
	
		
	i = 0
	while 1:
		pt = pt_list[i,:]
		print('pt: '+str(pt))
		DOT_IMG = copy.deepcopy(img)
		
		# Display blink dot
		ret_flag = blink_dot(DOT_IMG, (pt[1], pt[0]), 5)

		
		key = cv.WaitKey(500)		
		i=i+1
		
		if(ret_flag == 1):
			break
		
		if(key == 27):
			break

		if(i == pt_list.shape[0]):
			break;
			
	cv.DestroyWindow("CALIB_SCREEN")	


def blink_dot(img, pt, num_blinks):
	print('In Blink Dot')
	
	for i in range(0,num_blinks):
		cv2.circle(img, pt, 11, cv.Scalar(255, 255, 255), -1)
		cv2.imshow("CALIB_SCREEN", img)
		key = cv.WaitKey(500)
		
		img = np.zeros( img.shape)
		cv2.imshow("CALIB_SCREEN", img)
		key = cv.WaitKey(500)
		
		if(key == 27):
			return 1

		#READ_FLAG, RGB_img = vid_capt.read()
		#print(READ_FLAG)
	    #vid_w.write(RGB_img)
		#cv2.imwrite('hello.png', RGB_img)
	
## MAIN

if __name__ == '__main__':

	#CAM_CAPT = cv.CaptureFromCAM(0)
    #cv.SetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FPS, 30)
	#vid_capt = cv2.VideoCapture(0)
	#vid_capt.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	#vid_capt.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
	#size = (int(vid_capt.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(vid_capt.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
	
	#vid_w = cv2.VideoWriter('out_video.avi', cv.CV_FOURCC('M', 'J', 'P', 'G'), 30, size)
	#if(vid_w.isOpened() == False):
    #	print('Video file not opened!!')
	
	M = 1080
	N = 1920
	SCREEN = np.zeros( (M, N)) # Black Screen

	cv2.namedWindow("CALIB_SCREEN", cv.CV_WINDOW_NORMAL)
	cv2.moveWindow("CALIB_SCREEN", 0, 0)
	cv2.setWindowProperty("CALIB_SCREEN", cv2.WND_PROP_FULLSCREEN, cv.CV_WINDOW_FULLSCREEN)

	cv2.imshow("CALIB_SCREEN", SCREEN)

	#display_grid(SCREEN)
	display_rand_pattern(SCREEN)

	#vid_capt.release()
	#vid_w.release()

