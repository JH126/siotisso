from django.shortcuts import render
from django.http import HttpResponse
from subprocess import Popen, PIPE
import os
import argparse
import sys
import glob

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('origin_path')
parser.add_argument('folder_name')
parser.add_argument('file_name')
args = parser.parse_args()
origin_path = args.origin_path
folder_name = args.folder_name
file_name = (args.file_name).split('.')[0]


#frame divide
day_folder = '/testing/Images/' + folder_name + '/'
os.makedirs(day_folder, exist_ok=True)

own_folder = day_folder + file_name + '/'
frame_result_path = own_folder + 'frames/'

python_bin = '/virtualenvs/plate/Scripts/python.exe'
script_file = '/testing/Detection/Capture/video2frames.py'

p = Popen([python_bin, script_file, origin_path, frame_result_path, '--maxframes=15'],
            stdout=PIPE, encoding='utf8')

out, err = p.communicate()


#car detection
script_file = '/testing/Detection/car/Detect_Vehicle.py'

cars_folder = own_folder + 'cars/'
os.mkdir(cars_folder)

car_weight_path = '/testing/Detection/car/weight'
for i in range (15): #maybe modify
    frame_input_path = frame_result_path + 'frame_' + str(i) + '.jpg'
    save_path = cars_folder + 'car' + str(i).zfill(2) + '/'

    p = Popen([python_bin, script_file, '-i=' + frame_input_path, '-y='+car_weight_path, '-s='+save_path],
              stdout=PIPE, encoding='utf8')
    out, err = p.communicate() #synchronize
    


#plate detection
script_file = '/testing/Detection/LP/without_ocr.py'
plates_folder = own_folder + 'plates/'

lp_weight_path = '/testing/Detection/LP/yolo-coco'

for car_path in glob.glob(cars_folder + 'car*'):
    i = car_path[len(car_path)-2:len(car_path)]
    for input_car in glob.glob(car_path + '/result*.jpg'):
        save_path = plates_folder + 'car' + str(i) + '/'

        p = Popen([python_bin, script_file, '--image='+input_car, '-y='+lp_weight_path, '--save='+save_path],
                  stdout=PIPE, encoding='utf8')
        out, err = p.communicate() #synchronize


#super resolution
script_file = '/testing/srgan/srgan_test.py'
gans_folder = own_folder + 'srgan_results/'

sr_weight_path = '/testing/srgan/weights/gan_generator_1800_100.h5'

for plate_path in glob.glob(plates_folder + 'car*'):
    i = plate_path[len(plate_path)-2:len(plate_path)]
    
    for input_plate in glob.glob(plate_path + '/plate*.jpg'):
        j = input_plate[len(input_plate)-6:len(input_plate)-4]

        srgan_folder = gans_folder + 'car' + str(i) + '/' #time/srgan_results/car00/
        os.makedirs(srgan_folder, exist_ok = True)
        save_path = srgan_folder + 'srgan' + str(j) + '.jpg' #time/srgan_results/car00/srgan00.jpg
        print('save_path = ', save_path)
        p = Popen([python_bin, script_file, input_plate, save_path, sr_weight_path],
                  stdout=PIPE, encoding='utf8')
        out, err = p.communicate() #synchronize


#predict
script_file = '/testing/recognition/predict.py'
recog_folder = own_folder + 'recog/'

recog_weight_path = '/testing/recognition/saved_models/weights_best.pb'
save_folder = '/apps/project/static/result/'+ folder_name +'/' + file_name + '/'
os.makedirs(save_folder, exist_ok = True)

cnt = 0
flag = False #check predicted file existing
for gan_path in glob.glob(gans_folder + 'car*'): 
    i = gan_path[len(gan_path)-2:len(gan_path)]
    
    for input_gan in glob.glob(gan_path + '/srgan*.jpg'): 
        save_name = save_folder + 'result' + str(cnt)
        
        p = Popen([python_bin, script_file, '-i='+input_gan, '-s='+save_name, '-w='+recog_weight_path],
                  encoding='utf8')
        
        out, err = p.communicate() #synchronize
        cnt += 1
        flag = True


#save the result file
f = open("/apps/project/static/result.txt", "w")
if (flag):
    f.write(save_folder)
else :
    f.write('Nothing')
f.close()
