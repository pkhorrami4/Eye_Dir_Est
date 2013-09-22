import numpy as np
import cv2
import cv2.cv as cv
import copy
import math
from collections import deque


theta_range_lookup = {0:(0,90), 1:(0,180), 2:(90,180), 3:(-90,90), 4:(-180,180), 5:(90,270), 6:(-90,0), 7:(-180,0), 8:(-180,-90)}

def build_ind_map(M,N):
	
	alpha = 0.2
	
	ind_map = np.zeros( (M, N) )
	ind_map[0:(alpha*M), 0:(alpha*N)] = 0
	ind_map[(alpha*M):((1-alpha)*M), 0:(alpha*N)] = 3
	ind_map[((1-alpha)*M):, 0:(alpha*N)] = 6
	ind_map[0:(alpha*M), (alpha*N):((1-alpha)*N)] = 1
	ind_map[(alpha*M):((1-alpha)*M), (alpha*N):((1-alpha)*N)] = 4
	ind_map[((1-alpha)*M):, (alpha*N):((1-alpha)*N)] = 7
	ind_map[0:(alpha*M), ((1-alpha)*N):] = 2
	ind_map[(alpha*M):((1-alpha)*M), ((1-alpha)*N):] = 5
	ind_map[((1-alpha)*M):, ((1-alpha)*N):] = 8
	
	#print(str(ind_map))
	return ind_map

	
def gen_random_pt(prev_pt):
		
	print('Hi')

	#print(THETA_IND_MAP[prev_pt[1], prev_pt[0]])	
	theta_ind = THETA_IND_MAP[prev_pt[1], prev_pt[0]]
	theta_range = theta_range_lookup[theta_ind]
	#g = raw_input('...')
		
	if(theta_ind != 4):
		THETA = np.random.uniform(theta_range[0], theta_range[1])
	else:
		THETA = np.random.uniform(-15,15)
	MAG = 10#np.random.uniform(0, 100)
	
	R = [ MAG*math.sin(THETA * (180./math.pi) ), MAG*math.cos(THETA* (180./math.pi) )]
	cur_pt = prev_pt + R
	
	if( cur_pt[0] < 0):
		cur_pt[0] = 0
	elif( cur_pt[0] >= N):
		cur_pt[0] = (N-1)
	
	if( cur_pt[1] < 0):
		cur_pt[1] = 0
	elif( cur_pt[1] >= M):
		cur_pt[1] = (M-1)
	
	# Random Multi-variate Gaussian to be used for random walk
	#MU = [0,0]
	#SIGMA = [[10,250], [250,10]]
	#R = np.random.multivariate_normal(MU, SIGMA)

	#Rx = np.random.uniform(-50,50)
	#Ry = np.random.uniform(-50,50)
	#R = np.array([Rx, Ry])
	return cur_pt
	
	
def display_pts(screen_in, q, P, IND):


	# Get New Point 
	print('POINT '+str(IND)+'\n')
	
	# If first point, start from center
	if(IND == 0):		
		pt = np.array([N/2, M/2])
		q.append(pt)
		cv2.circle(screen_in, (pt[0], pt[1]), 11, cv.Scalar(0, 255, 0), -1)
		return q, screen_in
	else:

		if(IND > P):
			#print('q before removal: '+str(q)+'\n')
			#q = q[1:]
			q.popleft()
			#print('q after removal: '+str(q)+'\n')
			
		#print('LENGTH - q =' +str(len(q))+'\n')
		
		pt = gen_random_pt(q[-1])
		#pt = q[-1]+R
		q.append(pt)
		for i in range(0, len(q)):
			#print(str(q[i])+' ')
			cv2.circle(screen_in, (int(q[i][0]), int(q[i][1])), 11, cv.Scalar(0, 255, 0), -1)
		
		#print('\n')
		
		return q, screen_in



	
## MAIN

if __name__ == '__main__':

	M = 1080
	N = 1920
	
	THETA_IND_MAP = build_ind_map(M, N)
	#g = raw_input('...')


	
	SCREEN = np.zeros( (M, N, 3) ) # Black Screen

	cv2.namedWindow("CALIB_SCREEN", cv.CV_WINDOW_NORMAL)
	cv2.moveWindow("CALIB_SCREEN", 0, 0)
	cv2.setWindowProperty("CALIB_SCREEN", cv2.WND_PROP_FULLSCREEN, cv.CV_WINDOW_FULLSCREEN)
	cv2.imshow("CALIB_SCREEN", SCREEN)
	
	#vid_capt = 0
	#vid_capt = cv2.VideoCapture(0)
	#vid_capt.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	#vid_capt.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
	#vid_capt.set(cv.CV_CAP_PROP_FPS,30)
	
	
	i = 0	
	key = 0
	#q = [ ]	
	q = deque([]) # queue to hold P most recent points
	P = 1# number of points the queue holds and are displayed on the screen
	f = open('./Disp_Pts/disp_pts.txt', 'w')
	
	#vid_w = cv2.VideoWriter('./Capt_Vids/out.avi', cv.CV_FOURCC('I', '4','2', '0'), 30, (640, 480), 1)
	#cv2.namedWindow("CAPT",1)
	while(key != 27):

		# Display Points
		q, SCREEN_RET = display_pts(copy.deepcopy(SCREEN), q, P, i)
		f.write(str(i)+': '+str(q[-1])+'\n')
		#cv2.imshow("CALIB_SCREEN", SCREEN)
		cv2.imshow("CALIB_SCREEN", SCREEN_RET)
		
		# Capture Image
		#READ_FLAG, capt_img = vid_capt_obj.read()
		#cv2.imwrite('./Capt_Imgs/IMG_'+str(i).zfill(3)+'.png', capt_img)
		i += 1
		#vid_w.write(capt_img)
		#cv2.imshow("CAPT", capt_img)
		key = cv.WaitKey(33)
		
	#vid_w.release()
	#del vid_w
	#cv2.destroyWindow("CAPT")		
	cv2.destroyWindow("CALIB_SCREEN")
	#vid_capt.release()
	f.close()
