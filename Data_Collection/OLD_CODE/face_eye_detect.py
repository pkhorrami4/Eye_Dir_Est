import cv
import cv2
import numpy as np
  
def main_loop():

    CAM_CAPT = cv.CaptureFromCAM(0)
    cv.SetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FPS, 30)
    #cv.SetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FRAME_WIDTH, 1080)
    #cv.SetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FRAME_HEIGHT, 1920)
    
    count_no_face = 0
    
    
    while True:
        # Get a fresh frame
        RGB_img = cv.QueryFrame(CAM_CAPT)
        
        # Image Properties
        #print('Image Size: ({},{})'.format(cv.GetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FRAME_HEIGHT), cv.GetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FRAME_WIDTH)))
        #print('FPS: {}'.format(cv.GetCaptureProperty(CAM_CAPT, cv.CV_CAP_PROP_FPS)))
        
        haarFace = cv.Load('haarcascade_frontalface_default.xml')
        
        #RGB_img = cv.fromarray(np.array(rgb[:,:,::-1]))
        allFaces = cv.HaarDetectObjects(RGB_img, haarFace, cv.CreateMemStorage(), scale_factor=1.1, min_neighbors=10, flags=0, min_size=(50,50))
        
        # Get confidences
        if(allFaces != []):
            count_no_face = 0
            print(allFaces)
            face_confid = [c for ((x,y,w,h), c) in allFaces]
       
            max_ind = np.argmax(face_confid)
            FINAL_FACE = allFaces[max_ind]
        
        
            x0 = FINAL_FACE[0][0]
            y0 = FINAL_FACE[0][1]
            w = FINAL_FACE[0][2]
            h = FINAL_FACE[0][3]
        
            # Show detected face
            print('Face Detected!!')
            cv.Rectangle(RGB_img, (x0, y0), (x0+w, y0+h), cv.RGB(0,0,255), 2)
            
            # Detect eyes only in given face region
            print('Face: '+str(FINAL_FACE))
            cropped_img = RGB_img[y0:y0+h, x0:x0+w]
            #cv.Smooth(cropped_img, cropped_img, cv.CV_GAUSSIAN, 15, 15)
            #print(cv.GetSize(cropped_img))
            #cv.ShowImage('crop', cropped_img)
            #cv.SaveImage('IMAGE.png', cropped_img)
            allEyes = detect_eyes(cropped_img)
            
            print('Eyes: '+str(allEyes))
            for eye in allEyes:
                eye = eye[0]
                eye=(eye[0]+x0, eye[1]+y0, eye[2], eye[3])
                cv.Rectangle(RGB_img, (eye[0], eye[1]), (eye[0]+eye[2], eye[1]+eye[3]), cv.RGB(255,0,0), 2)
            
            
        else:
            print('No Face!!')
            count_no_face+=1
         
        if(count_no_face >= 10):
            font = cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX, 1.25, 1.25, 0, 3)
            cv.PutText(RGB_img, "Hey YOU!! Pay Attention!!", (50, 150), font, cv.RGB(255,0,0))
        
        
        cv.ShowImage('both', RGB_img)
        if(cv.WaitKey(5) == 27): # If ESC key pressed
            break
            
        
        print('\n')

    #cv.SaveImage('OUT3.png', RGB_img)
        
def detect_eyes(img):            
        haarEyes = cv.Load('haarcascade_eye.xml')
        #haarEyes = cv.Load('haarcascade_eye_tree_eyeglasses.xml')
        allEyes = cv.HaarDetectObjects(img, haarEyes, cv.CreateMemStorage(), scale_factor=1.25, min_neighbors=10, flags=0, min_size=(10,10))
        
        print('All Eyes: '+str(allEyes))
        #print('LENGTH: '+str(len(allEyes)))
        
        if(allEyes != []):
        
            if(len(allEyes) == 1):
                return allEyes
        
            else:
                eye_confid = [c for ((x,y,w,h), c) in allEyes]
                eye_confid_inds = np.argsort(eye_confid)[::-1][:2]
        
                #print(eye_confid_inds)
                eye0 = allEyes[eye_confid_inds[0]]
                eye1 = allEyes[eye_confid_inds[1]]
                                
                return [eye0, eye1]
        
        return []
        
        
main_loop()
cv.DestroyWindow('both')
#freenect.close_device()