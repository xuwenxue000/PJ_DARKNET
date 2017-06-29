

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
import random
import math



def init_origin_img(year,business,begin,limit=100):
    targetpath = project_dir+"/Invoicedevkit"+"/Invoice2017/JPEGImages/"
    targetpath2 = project_dir+"/Invoicedevkit"+"/Invoice2017/JPEGImages2/"
    target_rect_path = project_dir+"/Invoicedevkit"+"/Invoice2017/JPEGImages_rect/"
    target_rect_path2 = project_dir + "/Invoicedevkit" + "/Invoice2017/JPEGImages_rec2/"
    if not os.path.exists(targetpath):
        os.makedirs(targetpath)
    if not os.path.exists(targetpath2):
        os.makedirs(targetpath2)
    if not os.path.exists(target_rect_path):
        os.makedirs(target_rect_path)
    if not os.path.exists(target_rect_path2):
        os.makedirs(target_rect_path2)
    datas = m_invoice_dao.query_data(begin,limit)
    if(datas):
        for invoice in datas:
            print(str(invoice.id).zfill(8))
            image_id = str(invoice.id).zfill(8)
            imgurl = invoice.file_src
            re_str = '.*/(.*)\.(.*)'
            re_pat = re.compile(re_str)
            search_ret = re_pat.search(imgurl)
            if search_ret:
                out_path = targetpath + '%s.%s' % (image_id, 'jpg')
                if not os.path.exists(out_path):
                    try:
                        request.urlretrieve(imgurl, out_path)
                    except Exception as err:
                        print(err)
                    finally:
                        pass
                    if os.path.exists(out_path):
                        img = cv2.imread(out_path)
                        cv2.imwrite(out_path, img)


def  get_init_classes():
    classes=[]
    class_config_file_path = os.path.join(project_dir, "data", "invoice.names")
    with open(class_config_file_path) as invoice_names:
        class_name = invoice_names.readline()
        while class_name:
            class_name = class_name.strip()
            classes.append(class_name)
            class_name = invoice_names.readline()
    return classes



data_maxnum = 10

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open(business+'devkit/'+business+'%s/Annotations2/%s.xml'%(year, image_id))
    out_file = open(business+'devkit/'+business+'%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #print(cls)
        #print(classes)
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


def img_turn(image_id,forward):
    print("img_turn:",image_id)
    base_path = os.path.join(project_dir, business + "devkit", business + str(year))
    origin_annotations_dir = os.path.join(base_path, "Annotations")
    target_annotations_dir = os.path.join(base_path, "Annotations2")
    if not os.path.exists(target_annotations_dir):
        os.makedirs(target_annotations_dir)
    origin_xml_path = os.path.join(origin_annotations_dir, image_id + '.xml')

    origin_img_dir = os.path.join(base_path, "JPEGImages")
    target_img_dir = os.path.join(base_path, "JPEGImages2")

    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir)
    origin_img_path = os.path.join(origin_img_dir, image_id + '.jpg')

    target_rect_img_dir = os.path.join(base_path, "JPEGImages_rec2")

    trun_param=[]
    turn_num=0
    forward = int(forward)
    forward_index = forward
    while turn_num < 4:
        new_id = image_id+str(forward_index)
        forward_file_path = os.path.join(target_annotations_dir,new_id + ".xml")
        forward_img_path =os.path.join(target_img_dir, new_id + '.jpg')
        if forward_index==forward:
            shutil.copy(origin_xml_path,forward_file_path)
            shutil.copy(origin_img_path,forward_img_path)
            with open(forward_file_path) as forward_xml:
                tree = ET.parse(forward_xml)
                root = tree.getroot()
                root.find("filename").text = new_id + ".jpg"
                tree.write(forward_file_path)
        else:
            pre_forward_index = forward_index-1
            if pre_forward_index<0:
                pre_forward_index = 3
            pre_forwardd_file_path = os.path.join(target_annotations_dir,image_id+str(pre_forward_index)+".xml")
            pre_img_path = os.path.join(target_img_dir, image_id+str(pre_forward_index) + '.jpg')

            shutil.copy(pre_forwardd_file_path, forward_file_path)
            with open(forward_file_path) as forward_xml:
                tree = ET.parse(forward_xml)
                root = tree.getroot()
                size = root.find("size")
                width = size.find("width").text
                height = size.find("height").text
                size = root.find("size")

                size.find("width").text = height
                size.find("height").text = width
                root.find("filename").text=new_id+".jpg"
                tree.write(forward_file_path)
                #print(pre_img_path)
                if not os.path.exists(forward_img_path):
                    img = cv2.imread(pre_img_path)
                    rotateImg = rotate_about_center(img,-90)
                    cv2.imwrite(forward_img_path,rotateImg)
                width = int(width)
                height = int(height)
                #img = cv2.imread(forward_img_path)
                for obj in root.iter("object"):
                    bndbox = obj.find("bndbox")


                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)
                    new_ymin = xmin
                    new_ymax = xmax
                    new_xmin = height - ymax
                    new_xmax = height -ymin

                    obj.find("forward").text = str(forward_index)
                    obj.find("name").text = obj.find("key").text+"_"+str(forward_index)
                    bndbox.find("xmin").text = str(new_xmin)
                    bndbox.find("ymin").text = str(new_ymin)
                    bndbox.find("xmax").text = str(new_xmax)
                    bndbox.find("ymax").text = str(new_ymax)

                    #name = obj.find('name').text
                    #color=(255,0,0)
                    #img = cv2.rectangle(img, (new_xmin, new_ymin), (new_xmax, new_ymax), color, 2)
                    #img = cv2.putText(img, name, (new_xmax, new_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                tree.write(forward_file_path)
                #cv2.imwrite(os.path.join(target_rect_img_dir,new_id+".jpg"), img);






        forward_index+=1
        if forward_index>3:
            forward_index=0
        turn_num+=1



    pass



def init_label(year,business,is_turn=True):

    base_path = os.path.join(project_dir, business + "devkit", business + str(year))
    origin_annotations_path = os.path.join(base_path,"Annotations")
    annotations_path = os.path.join(base_path,"Annotations2")
    xmls = os.listdir(origin_annotations_path)
    main_dir = os.path.join(base_path, "ImageSets", "Main")
    train_txt = os.path.join(main_dir, "train.txt")
    test_txt = os.path.join(main_dir, "test.txt")
    val_txt = os.path.join(main_dir, "val.txt")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    train_txt_file = open(train_txt, 'w')
    test_txt_file = open(test_txt, 'w')
    val_txt_file = open(val_txt, 'w')
    for xml in xmls:
        xml_path = os.path.join(origin_annotations_path, xml)
        with open(xml_path) as f_xml:
            tree = ET.parse(f_xml)
            root = tree.getroot()
            filename = root.find('filename').text
            image_id = filename[:-4]
            objects = root.find('object')
            forward= None
            if objects and len(objects) > 0 :
                for obj in root.iter("object"):
                    forward = obj.find("forward").text
                    break
            if forward and is_turn:
                img_turn(image_id,forward)
                pass
    xmls2 = os.listdir(annotations_path)
    for xml in xmls2:
        xml_path = os.path.join(annotations_path,xml)
        with open(xml_path) as f_xml:
            tree = ET.parse(f_xml)
            root = tree.getroot()
            filename = root.find('filename').text
            image_id = filename[:-4]
            objects = root.find('object')
            if objects and len(objects) > 0 and filename:
                r_num = random.randint(0, 10)
                if r_num>8:
                    test_txt_file.write(image_id + "\n")
                else:
                    train_txt_file.write(image_id + "\n")
                val_txt_file.write(image_id + "\n")
    train_txt_file.close()
    test_txt_file.close()
    val_txt_file.close()



    sets = [('2017', 'train'), ('2017', 'val'), ('2017', 'test')]
    for year, image_set in sets:
        if not os.path.exists(project_dir+"/"+business + 'devkit/' + business + '%s/ImageSets/Main' % (year)):
            os.makedirs(project_dir+"/"+business + 'devkit/' + business + '%s/ImageSets/Main' % (year))
        image_ids = open(project_dir+"/"+business+'devkit/'+business+'%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
        for image_id in image_ids:
            list_file.write(project_dir+'/'+business+'devkit/'+business+str(year)+'/JPEGImages2/%s.jpg\n'%(str(image_id)))
            convert_annotation(year, image_id)
        list_file.close()

def predict_list():
    targetpath = project_dir + "/Invoicedevkit" + "/Invoice2017/JPEGImages/"
    target_rect_path = project_dir + "/Invoicedevkit" + "/Invoice2017/JPEGImages_rect/"

    xml_dir = os.path.join(project_dir, "Invoicedevkit", "Invoice2017", "Annotations")

    predict_list_file_path = os.path.join(project_dir, "predict_list.txt")

    with open(predict_list_file_path, 'w') as file:
        img_list = os.listdir(targetpath)
        for img_name in img_list:
            img_file_path = os.path.join(targetpath, img_name)
            image_id = img_name[:-4]
            xml_file_path = os.path.join(xml_dir, image_id + ".xml")
            if not os.path.exists(xml_file_path):
                file.write(img_file_path + "\n")
                print("predict_list=="+image_id)


year = 2017
business = "Invoice"

labels_dir = project_dir + "/" + business + 'devkit/' + business + '%s/labels/' % (year)
labels2_dir = project_dir + "/" + business + 'devkit/' + business + '%s/labels2/' % (year)
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)
else:
    shutil.rmtree(labels_dir)
    os.makedirs(labels_dir)

if os.path.exists(labels2_dir):
    shutil.rmtree(labels2_dir)
    pass


classes = get_init_classes()
init_origin_img(year,business,0)
#init_label(year,business)
#init_label(year,business,is_turn=False)
shutil.copytree(labels_dir,labels2_dir)

#predict_list()



