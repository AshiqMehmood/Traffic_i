#****************************************openCV yolov3 video code**********************************
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import requests
from pprint import pprint
import datetime as date
from datetime import timedelta
import os

path_helmet_out = '/content/helmets'    
path_violation_out = '/content/violations'

try:
    os.mkdir(path_helmet_out)   #create folder for saving output
    os.mkdir(path_violation_out)   
except OSError as error:
    print(error)

video = cv2.VideoCapture('/content/vid2.mp4') #to pass video path.. for live camera put '0'

if (video.isOpened() == False):
  print('unable to read video')
  
# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('vid_output.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, size)


whT = 320 
confThreshold = 0.5 #threshold value to determine the confidence of predicting an object. (50%) 
nmsThreshold = 0.3 #more the value more accuracy for suppressing mulitple detection of same object

classesFile = 'obj.names'
classNames = []

with open (classesFile, 'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')

colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8') #pick random colors for each class
modelConfiguration  = 'obj.cfg'
modelWeights = 'obj_last.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


width = 416
height = 416
dsize = (width, height)


def findObjects(outputs, img):
  hT, wT, cT = img.shape
  p = int(wT//2)  #center x
  q = int(hT//2)  #center y
  filter_width = int(wT-15) - int(p*3/4)
  filter_height = hT
  start_point = (int(p*3/4), 0)
  end_point =  (int(wT-15), int(hT))
  cv2.rectangle(img, start_point, end_point,(0,255,0),2)      #draw filter window
  filter_window_bbox = [[p, 0 , filter_width, filter_height]] #bbox for filtering/cropping
  bbox = []
  labeled_bbox_values = []
  helmet_bbox = []  
  no_helmet_bbox = []   
  rider_bbox = []
  plate_bbox= []
  classIds = [] #contain all class ids
  confs = []    #confidence value => whenever a confident object is found, its entered into this list

  for output in outputs:             #selecting one row at a time from 300/85 matrix 
    for det in output:               #now taking one colume at a time from the selected row
      scores = det[5:]               #remove first 5 columns containing: (x,y, width,height,confidence)
      classId = np.argmax(scores)    #find max value from columns 6-85 (to identify which class is detected among the 80 trained classes)
      confidence = scores[classId]   #find column with corresponding max value
      if confidence > confThreshold: #check it it has >50% confidence
        w, h = int(det[2]*wT), int(det[3]*hT)                   #get width and height
        x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)   #get x,y coordinates     
        bbox.append([x,y,w,h])
        classIds.append(classId)
        confs.append(float(confidence))  
        label = classNames[classId]
        labeled_bbox_values.append([label, x, y, w, h ])
        
  for box in labeled_bbox_values:
        label, left, top, right, bottom = box    
        #print(label, left, top,right,bottom)
        if (label.lower() == 'rider'):
          #print('rider added')
          rider_bbox.append([left, top, right, bottom])
        elif (label.lower() == 'helmet'):
          #print('helmet added')
          helmet_bbox.append([left, top, right, bottom])
        elif (label.lower() == 'no_helmet'):
          no_helmet_bbox.append([left, top, right, bottom])
        elif (label.lower() == 'plate'):
          plate_bbox.append([left, top, right, bottom])  

  #------------------------------checking if rider wears helmet or not----------------------------
  rider_with_helmet = []        #list to store bbox of rider with helmet
  rider_without_helmet = []     ##list to store bbox of rider without any helmet
           
  for fwb in filter_window_bbox:
    for rb in rider_bbox:     #iterate over every rider...at each cycle 'rb' is one rider
      flag = 0
      dupeFlag = 0  
      for hb in helmet_bbox:  #loop over every 'helmet' detected to find if the selected wears helmet
        if (rb[0] < hb[0]) and (rb[1] < hb[1]):
          if (hb[0] + hb[2]) < (rb[0] + rb[2]) and (hb[1] + hb[3]) < (rb[1] + rb[3]):
            rider_with_helmet.append([rb[0], rb[1], rb[2], rb[3]])
            print('helmet detected on this rider !') 
            x,y,w,h = rb[0], rb[1], rb[2], rb[3]
            crop1 = img[y:y+h,x:x+w] #crops
            filename = 'helmet' + str(rb[0]+rb[1]+rb[2]+rb[3]) + '.jpg'
            #print(filename)
            dest = os.path.join(path_helmet_out, filename)  #optional
            cv2.imwrite(dest, crop1)
            #cv2_imshow(crop1)       
            flag = 1                #flags to show that helmet is detected for the selected rider
            break
            #cv2_imshow(img)

        
      if rb not in rider_with_helmet: #check if this rider 'rb' is mentioned in the new list or not
        rider_without_helmet.append(rb)  #if true, push it to list of riders without helmet
        #x,y,w,h = rb[0], rb[1], rb[2], rb[3]
        x, y, w, h = fwb[0], fwb[1], fwb[2],fwb[3]
        crop2 = img[y:y+h,x:x+w] #crops to desired object  
        #cv2_imshow(crop2)
        dupeFlag = 1 
        print('<<<---------------No helmet found ------------>>>')
        filename = str(rb[0]+rb[1]+rb[2]+rb[3]) + '.jpg'
        #print(filename)
        dest = os.path.join(path_violation_out, filename) 
        cv2.imwrite(dest, crop2)    
    
      if not flag and not dupeFlag:  #case when helmet is not detected but no_helmet class is detected            
        for nb in no_helmet_bbox: #loop over every 'no_helmet' detected to find if the selected wears helmet
          if (rb[0] < nb[0]) and (rb[1] < nb[1]):
            if (nb[0] + nb[2]) < (rb[0] + rb[2]) and (nb[1] + nb[3]) < (rb[1] + rb[3]):
              rider_without_helmet.append(rb)
              x,y,w,h = rb[0], rb[1], rb[2], rb[3]
              crop3 = img[y:y+h,x:x+w] #crops  
              cv2_imshow(crop3)
              filename = 'violation' + str(rb[0]+rb[1]+rb[2]+rb[3]) + '.jpg'
              #print(filename)
              dest = os.path.join(path_violation_out, filename) 
              cv2.imwrite(dest, crop3)         
              print('<<<<<<<-----Rider not wearing helmet------->>>>>')
              #cv2_imshow(img)


        

  #print(labeled_bbox_values)  
  #---------------> length of rider bbox,helmet bbox, license bbox is required 
  #to proceed 'Traffic violation Algorithm'
  
  indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)  #Non max suppression: to eliminate multiple layers of same object detection   
  #print(indices) #[[0]]  
  for i in indices:
    i = i[0]
    box = bbox[i]
    x,y,w,h = box[0], box[1], box[2], box[3]         #extracting values of x,y,width,height from bbox 
    color = [int(c) for c in colors[classIds[i]]]    #example color = (255,0,255) 
    #draw bbox around detected image
    cv2.rectangle(img,(x,y),(x+w,y+h),color,2) #values entered are (image,(Xcoord,Ycoord),(X+width,Y+width), color, thickness)  
                                                         
    #putlabel on image
    cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', 
                (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2) 
    
    #if required, return cropped image
    #crop = img[y:y+h,x:x+w] #crops the detected license plate
    #print(box)
    #print(crop)
    #return crop
  
  #draw line for filtering image to get best results
  #cv2.line(img, (50,0), (50, 700), (0,255,0), 3)
  #cv2.line(img, (width-50,0), (width-50, 700), (0,255,0), 3)
  
  #result.write(img)         #write frame to video file
  cv2_imshow(img)#------------------------------------------------------------------->>>>> uncomment this to show frames on console 
 
  cv2.waitKey(1)   #put value 1 for video


layerNames = net.getLayerNames() #gets layer matrix
ln = net.getLayerNames()


while True: #--------------->> loop only for video !!!
  success, img = video.read()
  #img = cv2.resize(img, dsize)
  if success:
    try:
      blob = cv2.dnn.blobFromImage(img,1/255.0, (whT,whT),[0,0,0],1,crop=False) #cap >> img on video input
      net.setInput(blob)
      #print(net.getUnconnectedOutLayers())
      outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] #get the layer Name
      #print(outputNames)
      outputs = net.forward(outputNames)
      #print(outputs[0].shape) # (300, 85) => first layer produces 300 BBoxes
      #print(outputs[1].shape) # (1200, 85) => second layer '' 1200 ''
      #print(outputs[2].shape) # (4800, 85) => third layer '' 4800 ''
      #print(outputs[0][0])
      #cv2.line(img, (50,0), (50, height), (0,255,0), 4)
      #cv2.line(img, (width-50,0), (width-50, height), (0,255,0), 4) 
      
      findObjects(outputs, img) #call function to detect object
       
      if cv2.waitKey(1) & 0xFF == ord('s'):   #exit video on end
            break

    except Exception as e:
        print(str(e))
  else:
    break     

#-------------------------------------------------------------------------------------
def fetch_mock_DB(plate):
  dict_sample = {
          1:  {'Name' : 'Nicholas', 'Plate_no' : 'KLcc1234'},
          2:  {'Name' : 'Maverik', 'Plate_no' : 'KLab4445' },
          3:  {'Name' : 'Jocovic', 'Plate_no' : 'KLdj1467' },
          4:  {'Name' : 'harish', 'Plate_no' : 'MXcc2222' },
          5:  {'Name' : 'Pohikes', 'Plate_no' : 'NYmn6626' },
          6:  {'Name' : 'Girish', 'Plate_no' : 'KL01cn2774' }, 
          7:  {'Name' : 'Ragav', 'Plate_no' : 'TNbf3421' },
          8:  {'Name' : 'Joe', 'Plate_no' : 'KL22c9113' }, 
          9:  {'Name' : 'kayce', 'Plate_no' : 'KLyy6473' }, 
          10: {'Name' : 'Banega', 'Plate_no' : 'KL01F810' },
          11: {'Name' : 'Otomann', 'Plate_no' : 'kl22c3113' },  
          12: {'Name' : 'Alice', 'Plate_no' : 'UKPl8TE' }, 
          13: {'Name' : 'Harry', 'Plate_no' : 'GHmm3211' }, 
          14: {'Name' : 'sabari', 'Plate_no' : 'KL18g665' }, 
          15: {'Name' : 'Boston', 'Plate_no' : 'KLaf1001'},
          16: {'Name' : 'Sam', 'Plate_no': 'DL223401'},
          17: {'Name' : 'Yalie', 'Plate_no' : 'KL2c3113' },
          
    }

  present_time = date.datetime.now() #calculate  present time 

  for x in list(dict_sample):  
    if dict_sample[x].get('Plate_no').lower() == plate.lower(): #check if plate no is same
      driver = dict_sample[x].get('Name')
      dict_sample[x]['helmet'] =  'No'
      dict_sample[x]['Fee'] = '1000'
      due_date = present_time + timedelta(days=30)  #add 30 days from now
      dict_sample[x]['Due Date'] = str(due_date.strftime("%d %B %y"))
      print('Traffic Violation', dict_sample[x])
      with open('/content/file_out.txt', "a") as out: #open file in append mode
        out.write(str(dict_sample[x]) + "\n")         #write contents to file
      

#---------------------------------------------------------------------------------------2   
def detect_plate_ocr():
  for filename in os.listdir(path_violation_out):
    try:
      filepath = os.path.join(path_violation_out, filename)
      with open(filepath, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            #data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            headers={'Authorization': 'Token d66769a6127570550479f06569061cdcaf08dd65'})
      #pprint(response.json())

      a = response.json().get('results')  #get results from processing
      detected_plate = a[0].get('plate')  #obtain plate number from object

      if a:
        #print('License plate number: ',a[0].get('plate'))
        print('License Plate: ', detected_plate)      
        fetch_mock_DB(str(detected_plate)) #send plate number to mock-db for cross-checking
      else:
        print('Unable to Identify Plate !')
    
    except Exception as e:
      print(str(e))


video.release()
result.release()
cv2.destroyAllWindows()

detect_plate_ocr()
print("The video was successfully saved")

