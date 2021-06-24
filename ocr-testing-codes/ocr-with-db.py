#OCR to predict license plate
import requests
from pprint import pprint
import datetime as date
from datetime import timedelta

def fetch_mock_DB(plate):
  dict_sample = {
        1: {'Name' : 'Nicholas', 'Plate_no' : 'KLcc1234'},
        2: {'Name' : 'Maveric', 'Plate_no' : 'KLab4445' },
        3: {'Name' : 'Jocovic', 'Plate_no' : 'KLdj1467' },
        4: {'Name' : 'harish', 'Plate_no' : 'MXcc2222' },
        5: {'Name' : 'Pilo', 'Plate_no' : 'NYmn6626' },
        6: {'Name' : 'Girish', 'Plate_no' : 'KLaf1001' }, 
        7: {'Name' : 'Boston', 'Plate_no' : 'kl01cn2774'},
        
  }

  present_time = date.datetime.now()

  for x in list(dict_sample):  
    if dict_sample[x].get('Plate_no').lower() == plate.lower():
      driver = dict_sample[x].get('Name')
      dict_sample[x]['helmet'] =  'No'
      dict_sample[x]['Fee'] = '1000'
      due_date = present_time + timedelta(days=30)
      dict_sample[x]['Due Date'] = str(due_date.strftime("%d %B %y"))
      print('Traffic Violation', dict_sample[x])


with open('/content/b10.jpg', 'rb') as fp:
    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        #data=dict(regions=regions),  # Optional
        files=dict(upload=fp),
        headers={'Authorization': 'Token d66769a6127570550479f06569061cdcaf08dd65'})
#pprint(response.json())

a = response.json().get('results')
detected_plate = a[0].get('plate')

if a:
  #print('License plate number: ',a[0].get('plate'))
  fetch_mock_DB(str(detected_plate)) ##----------------------fetch details from mock db by passing plate 
else:
  print('Unable to Identify Plate !')
 


#mock_DB('kldj1467')
