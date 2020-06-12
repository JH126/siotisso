from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
import datetime
import subprocess
import json
from collections import OrderedDict
import os
from pathlib import Path
import glob

# Create your views here.

def index(request):

    return render(request, 'restoration/website.html')

def results(request): 
    
    if request.is_ajax and request.method == 'POST':
        uploaded_file = request.FILES['upload']
        uploaded_file_name = str(uploaded_file).split('.')
        
        file_format = uploaded_file_name[len(uploaded_file_name) -1]
        now = datetime.datetime.now()   #get time
        folder_name = now.strftime('%Y-%m-%d')  #set folder name
        file_name = now.strftime('%H%M%S_%f') + '.' + file_format    #set file name
        
        fs = FileSystemStorage(location='media/' + folder_name) #set route
        fs.save(file_name, uploaded_file) #save file
        file_path = '/apps/project/media/' + folder_name + "/" + file_name

        p = subprocess.Popen(['/virtualenvs/benedict/Scripts/python.exe',
                          '/testing/main.py', file_path, folder_name, file_name], stdout=subprocess.PIPE)
        out, err = p.communicate()
        
    return HttpResponse(data)
    

def check(request):
    
    if request.is_ajax:
        result_file = Path("/apps/project/static/result.txt")
        if result_file.is_file():
            f = open(result_file, "r")
            file_path = f.read()
            f.close()
            os.remove(result_file)
            
            if (file_path == 'Nothing'):
                return HttpResponse("Nothing")
            cnt = 0
            img_path = ''
            txt = ''
            for txt_path in glob.glob(file_path + '/result*.txt'):
                img_path += file_path.split("project/")[1] + 'result' + str(cnt) + '.jpg\n'
                f = open(txt_path,"r")
                txt += f.read()
                f.close()
                txt += "\n"
                cnt += 1

            data = img_path + '$$' + txt + '$$' + str(cnt)
            
            return HttpResponse(data)
        else:
            return HttpResponse("false")
    
