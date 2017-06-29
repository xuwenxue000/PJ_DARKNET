
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import re
import shutil
from urllib import request
import cv2
import sys
import numpy as np
py_dir = os.path.dirname(os.path.realpath(__file__))
project_dir= py_dir
sys.path.append(project_dir)
import invoice.dao.invoice_dao as m_invoice_dao
#project_dir = os.path.dirname(py_dir)
import math

targetpath = project_dir + "/Invoicedevkit" + "/Invoice2017/JPEGImages/"
target_rect_path = project_dir + "/Invoicedevkit" + "/Invoice2017/JPEGImages_rect/"

xml_dir = os.path.join(project_dir ,"Invoicedevkit","Invoice2017","Annotations")

predict_list_file_path = os.path.join(project_dir, "predict_list.txt")

with open(predict_list_file_path,'w') as file:
    img_list = os.listdir(targetpath)
    for img_name in img_list:
        img_file_path = os.path.join(targetpath,img_name)
        image_id = img_name[:-4]
        xml_file_path = os.path.join(xml_dir,image_id+".xml")
        #if not os.path.exists(xml_file_path):
        file.write(img_file_path+"\n")
        print(image_id)



