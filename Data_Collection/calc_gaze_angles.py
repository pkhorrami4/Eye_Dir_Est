import numpy as np
import cv2
import cv2.cv as cv
import copy
import glob
import os
import sys, getopt
import time


def compute_gaze_dir_vect(eye_coords, head_pose_angles, true_gaze_pt):

	# Princ. Pt.
	u_0 = 320
	v_0 = 240

	# Camera Focal Length (in pix.)
	f = 600 
	
	# Population Avg. Distance between Eyes (in mm)
	X_e = 62.5
	
	# Monitor Dimensions (in mm)
	MONITOR_DIM_X = 513
	MONITOR_DIM_Y = 348
	# Monitor Dimensions (in pix.)
	MONITOR_DIM_X_p = 1920
	MONITOR_DIM_Y_p = 1080	
	# mm/pix. ratio
	X_ratio = float(MONITOR_DIM_X) / MONITOR_DIM_X_p
	Y_ratio = float(MONITOR_DIM_Y) / MONITOR_DIM_Y_p
	
	
	# Compute 3D Coordinates of Subject's Eyes
	eye_R_img = eye_coords[0]
	eye_L_img = eye_coords[1]
	u_e = eye_L_img[0] - eye_R_img[0]
	print('u_e: '+str(u_e))
	Z = (-f*X_e )/(u_e)
	print('Z: '+str(Z))
	
	eye_R_WC = np.array([ (Z/-f)*(eye_R_img[0]-u_0), (Z/-f)*(eye_R_img[1]-v_0), Z])
	eye_L_WC = np.array([ (Z/-f)*(eye_L_img[0]-v_0), (Z/-f)*(eye_L_img[1]-v_0), Z])
	print('Right Eye World Coord: '+str(eye_R_WC))
	print('Left Eye World Coord: '+str(eye_L_WC))
	
	
	# Compute Pose Direction Vector
	alpha = head_pose_angles[0]
	beta = head_pose_angles[1]
	gamma = head_pose_angles[2]
	d_p = np.array([np.cos(alpha)*np.sin(beta), -np.sin(alpha), np.cos(alpha)*np.cos(beta)])	
	print('Head Pose Dir. Vector: '+str(d_p))
	
	# Convert True Gaze Point (to mm in monitor coordinate system)
	true_gaze_pt = true_gaze_pt * np.array([X_ratio, Y_ratio, 0]) - np.array([float(MONITOR_DIM_X)/2, float(MONITOR_DIM_Y)/2, 0])	
	true_gaze_pt_wcs = np.array([-true_gaze_pt[0], true_gaze_pt[1]+float(MONITOR_DIM_Y)/2 , 0])
	print('Gaze Point (mm - monitor_coord sys): '+str(true_gaze_pt))
	print('Gaze Point (mm - world coord sys): '+str(true_gaze_pt_wcs))
			
	# Compute Gaze Direction Vector
	d_g = (true_gaze_pt_wcs - eye_R_WC).astype('float32')/(-Z)
	print('Gaze Dir. Vector: '+str(d_g))
	
	
## MAIN

if __name__ == '__main__':
	print('Hello!')
	
	
	# Read in images from capt. list
	subj_num = 1 # will be passed as command line argument later
	img_path = './Capt_Imgs/Subj_'+str(subj_num)
	img_file_list = glob.glob('./Capt_Imgs/Subj_'+str(subj_num).zfill(3)+'/*.png')
	
		
	img_file_list = [img_file_list[0]] # for testing purporses
	for img_file_name in img_file_list:
		print(img_file_name+'\n')
		IMG = cv2.imread(img_file_name)
		
		## Read Output of Vuong's code to get Head Pose Angles and Eye Positions
		# Example
		eye_coords = [ np.array([268,200]), np.array([348,200]) ]
		head_pose_angles = [0,0,0]
		## END CALL VUONG'S CODE
		
		## Read in true gaze point on screen (in pix.)
		true_gaze_pt = np.array([1240, 193, 0])
		## END
		
		compute_gaze_dir_vect(eye_coords, head_pose_angles, true_gaze_pt)
		