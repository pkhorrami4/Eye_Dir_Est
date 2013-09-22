import numpy as np
import cv2
import cv2.cv as cv
import copy
import os
import sys, getopt
import time

def detect_face(img):	
	
	haarFace = cv.Load('./haarcascade_frontalface_default.xml')
        
	#RGB_img = cv.fromarray(np.array(rgb[:,:,::-1]))
	allFaces = cv.HaarDetectObjects(cv.fromarray(img), haarFace, cv.CreateMemStorage(), scale_factor=1.1, min_neighbors=3, flags=0, min_size=(50,50))

	# Get confidences
	if(allFaces != []):
		count_no_face = 0
		#print(allFaces)
		face_confid = [c for ((x,y,w,h), c) in allFaces]
   
		max_ind = np.argmax(face_confid)
		FINAL_FACE = allFaces[max_ind]
	
		return FINAL_FACE[0]
		
	else:
		return []
		
   
## MAIN

if __name__ == '__main__':

	print('Hello!')

	subj_num = 0
	if(len(sys.argv) != 3):
		print('disp_grid_w_vid_capt.py -s subj_num')
		sys.exit(1)
	
	try:
		opts, args = getopt.getopt(sys.argv[1:],"s:")
	except getopt.GetoptError:
		print('disp_grid_w_vid_capt.py -s subj_num')
		sys.exit(2)

	for opt, arg in opts:
		if(opt == '-s'):
			subj_num = arg

			
	if(os.path.exists('./Capt_Imgs/Subj_'+str(subj_num).zfill(3)+'/')!=True):
		os.mkdir('./Capt_Imgs/Subj_'+str(subj_num).zfill(3))
			
	g = raw_input('...')
	
	M = 1080
	N = 1920
	
	## File to store positions of poitns displayed	
	f = open('./Disp_Pts/Subj_'+str(subj_num).zfill(3)+'_disp_pts.txt', 'w')

	
	## Construct Black Screen
	SCREEN = np.zeros( (M, N, 3) ) 
	
	cv2.namedWindow("CALIB_SCREEN", cv.CV_WINDOW_NORMAL)
	cv2.moveWindow("CALIB_SCREEN", 0, 0)
	cv2.setWindowProperty("CALIB_SCREEN", cv2.WND_PROP_FULLSCREEN, cv.CV_WINDOW_FULLSCREEN)
	cv2.imshow("CALIB_SCREEN", SCREEN)
	
	# Construct grid overlay onto black screen
	NUM_COL_PTS = 24
	NUM_ROW_PTS = 14
	[col_ind, row_ind] = np.meshgrid( np.linspace(40, N-40, NUM_COL_PTS).astype('int'), np.linspace(40, M-40, NUM_ROW_PTS).astype('int')  )
	#print(str(col_ind)+'\n')
	#print(str(row_ind)+'\n')
	
	
	## Video Capture Object	
	#vid_capt = 0
	vid_capt = cv2.VideoCapture(0)
	vid_capt.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	vid_capt.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
	vid_capt.set(cv.CV_CAP_PROP_FPS,30)
	READ_FLAG, init_capt_img = vid_capt.read()
	#cv2.namedWindow("CAPT",1)
	
	img_ind = 0
	for r in range(NUM_ROW_PTS):
	
		vec_r = row_ind[r,:]
		vec_c = col_ind[r,:]
		# Create raster effect
		if( (r%2)==1): 
			vec_c = vec_c[::-1]
	
		for c in range(NUM_COL_PTS):
								
			f.write('('+str(row_ind[r,c])+', '+str(col_ind[r,c])+')\n')
			#print('('+str(vec_r[c])+' '+str(vec_c[c])+')\n')
			
			# Display Circle
			start_time = time.time()
			SCREEN_CIRC = np.zeros( SCREEN.shape)
			cv2.circle(SCREEN_CIRC, (vec_c[c], vec_r[c]), 11, cv.Scalar(0, 255, 0), -1)
			cv2.imshow("CALIB_SCREEN", SCREEN_CIRC)
			end_time = time.time()
			cv.WaitKey(300)
			end_time2 =time.time()
			print('Time to display: '+str(end_time-start_time)+'\n')
			#print('Time to display + 1 sec: '+str(end_time2-start_time)+'\n')
			
			# Capture Frame
			start_time_read = time.time()
			read_flag, capt_img = vid_capt.read()
			end_time_read = time.time()
			
			# Detect Face
			#FACE = detect_face(capt_img)   
			#print(str(FACE)+'\n')			
			
			img_ind += 1 
			start_time_write = time.time()			
			#if(FACE != []):
			#	capt_img = capt_img[FACE[1]:FACE[1]+FACE[3], FACE[0]:FACE[0]+FACE[2]]
			cv2.imwrite('./Capt_Imgs/Subj_'+str(subj_num).zfill(3)+'/IMG_'+str(img_ind).zfill(3)+'.png', capt_img)					
			end_time_write = time.time()
			print('Time to read: '+str(end_time_read-start_time_read)+'\n')
			print('Time to write: '+str(end_time_write-start_time_write)+'\n')
	
	f.close()	
	#cv2.destroyWindow("CAPT")		
	cv2.destroyWindow("CALIB_SCREEN")
	#vid_capt.release()	
	g = raw_input('...')
	

	
	#vid_w = cv2.VideoWriter('./Capt_Vids/out.avi', cv.CV_FOURCC('I', '4','2', '0'), 30, (640, 480), 1)
	#cv2.namedWindow("CAPT",1)
	#vid_w.write(capt_img)
	#vid_w.release()
	#del vid_w
	#cv2.destroyWindow("CAPT")		
	#cv2.destroyWindow("CALIB_SCREEN")
	#vid_capt.release()
	#f.close()
