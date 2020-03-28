import numpy as np
import time
import cv2
from pygame import mixer

def disp(window,img):
    cv2.namedWindow(window ,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1500,800)
    cv2.imshow(window,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

sounds={1:'C', 2:'C_s', 3:'D', 4:'D_s', 5:'E', 6:'F', 7:'F_s', 8:'G', 9:'G_s', 10:'A', 11:'Bb', 12:'B', 13:'C1', 14:'C_s1', 15:'D1', 16:'D_s1', 17:'E1'}
max_Keys_Allowed=17

#Verbose = True
Verbose = False

#blueLower = (80,150,50) #for blue
#blueUpper = (120,255,255)
blueLower = (0, 48, 80) #for skin
blueUpper = (20, 255, 255)


camera = cv2.VideoCapture(0)
ret,frame = camera.read()
H,W = frame.shape[:2]

key_leftmost= W*1//8
key_rightmost=W*7//8
key_topmost= H*2//8
key_bottommost=H*5//8

n=min(int(input("enter no. of keys (max="+str(max_Keys_Allowed)+")")),max_Keys_Allowed)
#n=17
key_width_extra=(key_rightmost-key_leftmost)%n
key_rightmost=key_rightmost-key_width_extra
key_width=(key_rightmost-key_leftmost)//n
key_height=key_bottommost-key_topmost
key_area=key_height*key_width
threshold_area_value=key_area*0.1*255
key= cv2.resize(cv2.imread('Images/OneWhiteKey.png'),(key_width,key_height),interpolation=cv2.INTER_CUBIC)
#disp("img",key)
#key.shape
PianoImage=np.tile(key,[1,n,1])
#PianoImage.shape
#disp("piano",PianoImage)
mixer.init()
#mixer.set_num_channels(n)
mixer.set_num_channels(n*50)
soundList=[mixer.Sound("../Music_Notes/"+sounds[i+1]+'.wav') for i in range(n)]
yellowTouch=np.zeros([key_height,key_width,3],dtype="uint8")
yellowTouch[:,:,0]=0
yellowTouch[:,:,1]=247
yellowTouch[:,:,2]=255
def key_Addr():
    keyAddr=[]
    prevLeft=key_leftmost
    for i in range(n):
        keyAddr.append([prevLeft,prevLeft+key_width])
        prevLeft=keyAddr[i][1]
    return keyAddr

def playKey(mask,currentPressedKeys):
    eachKeyMaskSum= np.sum(mask,axis=0).reshape(n,-1).sum(axis=1)
    keyToPlay=np.where(eachKeyMaskSum>=threshold_area_value)[0]
    newKeyPressed=np.setdiff1d(keyToPlay,currentPressedKeys,assume_unique=True)
    for i in newKeyPressed:
        print(i)
#        mixer.Channel(i).play(soundList[i])
        mixer.find_channel().play(soundList[i])
    return keyToPlay
#        time.sleep(0.001)
#        x=np.zeros([10,10])
#        a=np.array([1,5,8 ,9],dtype="int")
#        np.where(a>2)[0]
#        b=np.array([2,6],dtype="int")
#        x[a,:]
#        x[a:b,:]  
#    mixer.set_num_channels(10000)
#    c=0 
#    for j in range(0,100000000):
#        for i in range(0,15):
#            c=c+1
#            print(c)
#            print(i)
#            mixer.find_channel().play(soundList[i])
#            mixer.Channel(i).play(soundList[i])
#            sleep(1)
def ROI_analysis(ROIregion):
    hsv = cv2.cvtColor(ROIregion, cv2.COLOR_BGR2HSV)
    # generating mask for pixels corresponding to detected blue colour.
    mask= cv2.inRange(hsv, blueLower, blueUpper)
    return mask
currentPressedKeys=[]
keyAddr=key_Addr()
cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output',1800,1000)

def getPressedPianoMask(currentPressedKeys):
    pressedPianoMask=np.copy(PianoImage)
    for i in currentPressedKeys:
        pressedPianoMask[:,i*key_width:(i+1)*key_width,:] = cv2.addWeighted(pressedPianoMask[:,i*key_width:(i+1)*key_width,:], 0.8,np.copy(yellowTouch), 8, 0)
#        pressedPianoMask[:,i*key_width:(i+1)*key_width,0]=pressedPianoMask[:,i*key_width:(i+1)*key_width,0]//10
#        pressedPianoMask[:,i*key_width:(i+1)*key_width,1]=(pressedPianoMask[:,i*key_width:(i+1)*key_width,0]+247)//2
#        pressedPianoMask[:,i*key_width:(i+1)*key_width,2]=(pressedPianoMask[:,i*key_width:(i+1)*key_width,0]+255)//2
#    disp("d",pressedPianoMask)
    return pressedPianoMask
currentPressedKeys=[]
while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame,1)
    #print("hello")
    #disp("frame",frame)
    if not(ret):
        print("unable to access camera")
        break

    ROIregion= np.copy(frame[key_topmost:key_bottommost,key_leftmost:key_rightmost])
    #disp("ROIregion",ROIregion)
    mask=ROI_analysis(ROIregion)
    #disp("as",mask)=ROI_analysis(ROIregion)
    #mask=np.random.randint(0,high=2, size=(6,10))
    sumation = np.sum(mask)
    if sumation >= threshold_area_value:
        currentPressedKeys=playKey(mask,currentPressedKeys)
    else:
        currentPressedKeys=[]
    pressedPianoMask=getPressedPianoMask(currentPressedKeys)
    # A writing text on an image.
    #cv2.putText(frame,'Project: Air Drums',(10,30),2,1,(20,20,20),2)
    # Display the ROI to view the blue colour being detected
    if Verbose:
        frame[key_topmost:key_bottommost,key_leftmost:key_rightmost]=cv2.bitwise_and(frame[key_topmost:key_bottommost,key_leftmost:key_rightmost],frame[key_topmost:key_bottommost,key_leftmost:key_rightmost],mask=mask)
        frame[key_topmost:key_bottommost,key_leftmost:key_rightmost] = cv2.addWeighted(pressedPianoMask, 0.2, frame[key_topmost:key_bottommost,key_leftmost:key_rightmost], .8, 0)                
    else:
        frame[key_topmost:key_bottommost,key_leftmost:key_rightmost]=cv2.bitwise_and(frame[key_topmost:key_bottommost,key_leftmost:key_rightmost],frame[key_topmost:key_bottommost,key_leftmost:key_rightmost],pressedPianoMask,mask=mask)
#        frame[key_topmost:key_bottommost,key_leftmost:key_rightmost] = cv2.addWeighted(pressedPianoMask, 0.5, frame[key_topmost:key_bottommost,key_leftmost:key_rightmost], 1, 0)
    #disp("frame",frame)
    cv2.imshow('Output',frame)
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed== ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

