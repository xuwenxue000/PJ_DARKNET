# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets,QtGui,QtCore
import sys
import os
import shutil
import cv2
import xml.etree.ElementTree as ET
import math
import time
from threading import Timer
import re
py_path = os.path.dirname(__file__)
project_path = py_path


class ImageInfo:
    def __init__(self):
        self.rects={}
        self.current_rect_key=None





class MyWindow(QtWidgets.QMainWindow):


    def config(self,last_dir=0):
        self.default_dir = ""
        self.dir_list = []

        xml_path = os.path.join(project_path, "markutil.conf")
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                if last_dir:
                    root.find("last_dir").text=last_dir
                self.default_dir = root.find("last_dir").text
                for dir in root.find("dir_list").iter('dir'):
                    self.dir_list.append(dir.text)




    def config_devkit(self,file_dir,image_id=0):
        xml_path = os.path.join(file_dir, "markutil.conf")
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                if image_id:
                    root.find("last_index").text = image_id


    def get_config_devkit_last_index(self,file_dir):
        xml_path = os.path.join(file_dir, "markutil.conf")
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                last_index = root.find("last_index").text
                if not last_index or last_index=="":
                    last_index = os.listdir(os.path.join(file_dir,"JPEGImages"))[0][:-4]
                    root.find("last_index").text =last_index
                    tree.write(xml_path)
                return last_index

    def get_config_devkit_mark_key(self,file_dir):
        self.mark_key = []
        xml_path = os.path.join(file_dir, "markutil.conf")
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                mark_key = root.find("mark_key").text
                with open(os.path.join(project_path, mark_key)) as mark_key_file:
                    line = mark_key_file.readline();
                    while line and line != "":
                        self.mark_key.append(line)
                        line = mark_key_file.readline();


    def __init__(self):
        super(MyWindow, self).__init__()
        self.change_mode = 1# 0是移动,1是改动
        self.change_step = 5
        self.config()
        self.image_info = ImageInfo()
        self.resize(500, 300)
        grid = QtWidgets.QGridLayout()
        widget = QtWidgets.QWidget()
        widget.setFixedWidth(1024)
        widget.setFixedHeight(700)
        widget.setLayout(grid)
        self.setCentralWidget(widget)
        #qr = self.frameGeometry()
        #cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        #qr.moveCenter((0,0))
        self.move(QtCore.QPoint(0,0))

        comb_forward = QtWidgets.QComboBox()
        comb_forward.addItem("0")
        comb_forward.addItem("1")
        comb_forward.addItem("2")
        comb_forward.addItem("3")
        comb_forward.setCurrentIndex(0)
        comb_forward.setFixedWidth(50)
        comb_forward.currentIndexChanged.connect(self.forward_change)
        self.comb_forward = comb_forward
        grid.addWidget(comb_forward, 0, 0)


       # grid.addWidget(QtWidgets.QLabel("图片路径:"), 0, 0)

        #default_file_dir = "Invoicedevkit/Invoice2017/JPEGImages"
        default_file_dir = self.default_dir
        self.get_config_devkit_mark_key(default_file_dir)
        comb_file_dir = QtWidgets.QComboBox()
        for file_dir in self.dir_list:
            comb_file_dir.addItem(file_dir)
        comb_file_dir.setCurrentText(default_file_dir)
        comb_file_dir.setFixedWidth(300)
        self.comb_file_dir = comb_file_dir
        grid.addWidget(comb_file_dir, 0, 1)


        #edit_file_dir = QtWidgets.QLineEdit()
        #edit_file_dir.setMaximumWidth(500)
        #self.edit_file_dir=edit_file_dir
        #edit_file_dir.setText(default_file_dir)
        #grid.addWidget(edit_file_dir, 0, 1)

        edit_file_name = QtWidgets.QLineEdit()
        default_file_name = self.get_default_image_id()
        edit_file_name.setText(default_file_name)
        edit_file_name.setMaximumWidth(200)
        self.edit_file_name=edit_file_name
        #widget.connect(self.edit_file_path, QtCore.SIGNAL("returnPressed()"), self.updateUi)
        grid.addWidget(edit_file_name,1,1)


        edit_setp = QtWidgets.QLineEdit()
        edit_setp.setText('1')
        edit_setp.setMaximumWidth(200)
        self.edit_setp = edit_setp
        # widget.connect(self.edit_file_path, QtCore.SIGNAL("returnPressed()"), self.updateUi)
        grid.addWidget(edit_setp, 2, 1)

        button_read_img = QtWidgets.QPushButton()
        button_read_img.setText("读取")
        button_read_img.clicked.connect(self.read_img)
        #grid.addWidget(button_read_img, 0, 3)


        button_pre_img = QtWidgets.QPushButton()
        button_pre_img.setText("上一张")
        button_pre_img.clicked.connect(self.pre_img)
        button_pre_img.setFixedWidth(100)
        grid.addWidget(button_pre_img, 1, 0)

        button_next_img = QtWidgets.QPushButton()
        button_next_img.setText("下一张")
        button_next_img.clicked.connect(self.next_img)
        button_next_img.setFixedWidth(100)
        grid.addWidget(button_next_img, 2, 0)



        button_up_rect = QtWidgets.QPushButton()
        button_up_rect.setText("上")
        button_up_rect.clicked.connect(self.up_rect)
        button_up_rect.setFixedWidth(50)
        grid.addWidget(button_up_rect, 0, 3)


        button_move_rect = QtWidgets.QPushButton()
        button_move_rect.setText("to移")
        button_move_rect.clicked.connect(self.change_rect)
        button_move_rect.setFixedWidth(50)
        grid.addWidget(button_move_rect, 1, 3)
        self.button_move_rect=button_move_rect



        button_down_rect = QtWidgets.QPushButton()
        button_down_rect.setText("下")
        button_down_rect.clicked.connect(self.down_rect)
        button_down_rect.setFixedWidth(50)
        grid.addWidget(button_down_rect, 2, 3)

        button_left_rect = QtWidgets.QPushButton()
        button_left_rect.setText("左")
        button_left_rect.clicked.connect(self.left_rect)
        button_left_rect.setFixedWidth(50)
        grid.addWidget(button_left_rect, 1, 2)
        button_right_rect = QtWidgets.QPushButton()
        button_right_rect.setText("右")
        button_right_rect.clicked.connect(self.right_rect)
        button_right_rect.setFixedWidth(50)
        grid.addWidget(button_right_rect, 1, 4)






        button_up_up_rect = QtWidgets.QPushButton()
        button_up_up_rect.setText("上下")
        button_up_up_rect.clicked.connect(self.up_down_rect)
        button_up_up_rect.setFixedWidth(50)
        grid.addWidget(button_up_up_rect, 0, 4)

        button_down_up_rect = QtWidgets.QPushButton()
        button_down_up_rect.setText("下上")
        button_down_up_rect.clicked.connect(self.down_up_rect)
        button_down_up_rect.setFixedWidth(50)
        grid.addWidget(button_down_up_rect, 2, 2)

        button_left_right_rect = QtWidgets.QPushButton()
        button_left_right_rect.setText("左右")
        button_left_right_rect.clicked.connect(self.left_right_rect)
        button_left_right_rect.setFixedWidth(50)
        grid.addWidget(button_left_right_rect, 0, 2)

        button_right_left_rect = QtWidgets.QPushButton()
        button_right_left_rect.setText("右左")
        button_right_left_rect.clicked.connect(self.right_left_rect)
        button_right_left_rect.setFixedWidth(50)
        grid.addWidget(button_right_left_rect, 2, 4)


        comb_addkey = QtWidgets.QComboBox()
        for mark_k in self.mark_key:
            comb_addkey.addItem(mark_k.strip())
        #comb_addkey.addItem("car_engine")
        #comb_addkey.addItem("car_vin")
        #comb_addkey.addItem("car_price")
        #comb_addkey.addItem("car_organ")
        comb_addkey.setCurrentIndex(0)
        comb_addkey.setFixedWidth(150)
        #comb_addkey.currentIndexChanged.connect(self.add_change)
        self.comb_addkey = comb_addkey
        grid.addWidget(comb_addkey, 0, 5)

        delete_delete_rect = QtWidgets.QPushButton()
        delete_delete_rect.setText("删除")
        delete_delete_rect.clicked.connect(self.delete_rect)
        delete_delete_rect.setFixedWidth(50)
        grid.addWidget(delete_delete_rect, 1, 5)


        button_disable_img = QtWidgets.QPushButton()
        button_disable_img.setText("不可用")
        button_disable_img.clicked.connect(self.disable_img)
        button_disable_img.setFixedWidth(60)
        grid.addWidget(button_disable_img, 2, 5)

        label_img= MyLabel(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(label_img)
        #scroll.setAutoFillBackground(True)
        #scroll.setWidgetResizable(True)
        # grid.addWidget(scroll)
        #img_read = QtGui.QPixmap(os.path.join(project_path, default_file_dir,default_file_name+'.jpg'))
        #label_img.setPixmap(img_read)
        #label_img.setFixedWidth(1024)
        #label_img.setFixedHeight(768)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        grid.addWidget(scroll,3,0,1,7)
        self.label_img =label_img
        #grid.addWidget(label_img,3,0,1,6)
        self.read_img()
        self.statusChange()

    def get_current_file_dir(self):
        return self.comb_file_dir.currentText()

    def disable_img(self):
        current_name = self.image_info.current_rect_key
        file_dir = os.path.join(project_path, self.get_current_file_dir())
        image_id = self.edit_file_name.text()
        xml_path = os.path.join(file_dir, "Annotations", image_id + '.xml')
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    obj.find('difficult').text='1'
                tree.write(xml_path)
            return True
        else:
            return False

    def delete_rect(self):
        current_name = self.image_info.current_rect_key
        file_dir = os.path.join(project_path, self.get_current_file_dir())
        image_id = self.edit_file_name.text()
        xml_path = os.path.join(file_dir, "Annotations", image_id + '.xml')
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    name = obj.find('name').text
                    if name == current_name:
                        root.remove(obj)
                        tree.write(xml_path)
                        self.read_img()
                        break
            return True
        else:
            return False


    def get_default_image_id(self):
        xml_path = os.path.join(project_path, "markutil.conf")
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                last_dir = root.find('last_dir').text
                markutil_path = os.path.join(project_path, last_dir,"markutil.conf")
                return self.get_config_devkit_last_index(os.path.join(project_path, last_dir))

        return '00000001'


    def set_default_image_id(self):
        image_id = self.edit_file_name.text()
        last_dir = self.get_current_file_dir()
        xml_path = os.path.join(project_path, "markutil.conf")
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                root.find('last_dir').text=last_dir
                markutil_path = os.path.join(project_path, last_dir,"markutil.conf")
                if os.path.exists(markutil_path):
                    with open(markutil_path) as markutil_xml:
                        markutil_tree = ET.parse(markutil_xml)
                        markutil_root = markutil_tree.getroot()
                        markutil_root.find('last_index').text=image_id
                        markutil_tree.write(markutil_path)

    def statusChange(self):
        message="";
        if self.change_mode:
            message += "[选中区域修改]"
        else:
            message += "[选中区域移动]"

        file_dir = os.path.join(project_path, self.get_current_file_dir())
        image_id = self.edit_file_name.text()
        xml_path = os.path.join(file_dir, "Annotations", image_id + '.xml')
        difficult ='0'
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    if difficult == '1':
                        break
        if difficult == '1':
            message += "[此数据不可用]"
        self.statusBar().showMessage(message)

    def forward_change(self):
        self.save_xml_by_forward(self.comb_forward.currentText())
        self.read_img()

    def change_rect(self):
        mode=0
        text = 'to改'
        if not self.change_mode:
            mode = 1
            text = 'to移'
        self.change_mode = mode
        self.button_move_rect.setText(text)
        self.statusChange()
        self.read_img()
        pass

    def up_up_rect(self):
        self.change_step  = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        ymin = min(rect[1],rect[3])
        ymin -= self.change_step
        self.save_xml(rect_key,"ymin",ymin)
        self.read_img()


    def down_up_rect(self):
        self.change_step  = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        ymax = max(rect[1],rect[3])
        ymax -= self.change_step
        self.save_xml(rect_key,"ymax",ymax)
        self.read_img()
    def up_down_rect(self):
        self.change_step  = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        ymin = min(rect[1],rect[3])
        ymin += self.change_step
        self.save_xml(rect_key,"ymin",ymin)
        self.read_img()


    def up_rect(self):
        self.change_step  = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects.get(rect_key)
        if rect:
            ymin = rect[1]
            ymax = rect[3]

            if not self.change_mode:
                ymax -= self.change_step
                ymin -= self.change_step
            else:
                ymin -= self.change_step
            self.save_xml(rect_key,"ymin",ymin,"ymax",ymax)
            self.read_img()

    def down_rect(self):
        self.change_step = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        ymin = rect[1]
        ymax = rect[3]

        if not self.change_mode:
            ymin += self.change_step
            ymax += self.change_step
        else :
            ymax += self.change_step
        self.save_xml(rect_key, "ymin", ymin, "ymax", ymax)
        self.read_img()

    def left_rect(self):
        self.change_step = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        xmin = rect[0]
        xmax = rect[2]

        if not self.change_mode:
            xmin -= self.change_step
            xmax -= self.change_step
        else:
            xmin -= self.change_step
        self.save_xml(rect_key, "xmin", xmin, "xmax", xmax)
        self.read_img()
        pass

    def left_right_rect(self):
        self.change_step = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        xmin = min(rect[0],rect[2])
        xmin += self.change_step
        self.save_xml(rect_key, "xmin", xmin)
        self.read_img()
        pass

    def right_left_rect(self):
        self.change_step = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        xmax = max(rect[0],rect[2])
        xmax -= self.change_step
        self.save_xml(rect_key, "xmax", xmax)
        self.read_img()

    def right_rect(self):
        self.change_step = int(self.edit_setp.text())
        rect_key = self.image_info.current_rect_key
        rect = self.image_info.rects[rect_key]
        xmin = rect[0]
        xmax = rect[2]
        if not self.change_mode:
            xmin += self.change_step
            xmax += self.change_step
        else:
            xmax += self.change_step
        self.save_xml(rect_key, "xmin", xmin, "xmax", xmax)
        self.read_img()
        pass


    def save_xml(self,current_name,key,value,key2=0,value2=0):
        file_dir = os.path.join(project_path, self.get_current_file_dir())
        image_id = self.edit_file_name.text()
        xml_path = os.path.join(file_dir, "Annotations", image_id + '.xml')
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    name = obj.find('name').text
                    if name==current_name:
                        xmlbox.find(key).text =str(value)
                        if key2:
                            xmlbox.find(key2).text = str(value2)
                        tree.write(xml_path)
                        break
            return True
        else:
            return False
    def save_xml_by_forward(self,forward):
        file_dir = os.path.join(project_path, self.get_current_file_dir())
        image_id = self.edit_file_name.text()
        xml_path = os.path.join(file_dir, "Annotations", image_id + '.xml')
        if os.path.exists(xml_path):
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    key = obj.find('key').text
                    obj.find('forward').text = forward
                    name = key+"_"+forward
                    obj.find('name').text = name
                    tree.write(xml_path)
            self.read_img()
            return True
        else:
            return False

    def pre_img(self):
        file_dir = os.path.join(project_path, self.get_current_file_dir(), "JPEGImages")
        image_id = self.edit_file_name.text()
        file_list = os.listdir(file_dir)
        result_image_id = image_id
        file_index = 0
        for file_name in file_list:
            file_image_id = file_name[:-4]
            if (file_image_id == image_id):
                result_image_id = file_list[file_index-1][:-4]
                break
            file_index += 1

        self.edit_file_name.setText(result_image_id)
        self.image_info.current_rect_key = None

        if not self.read_img():
            self.edit_file_name.setText(image_id)
        else:
            self.set_default_image_id()
            self.image_info.current_rect_key = None
            self.comb_addkey.setCurrentText('car_idcard')
            self.comb_addkey.setCurrentIndex(0)
            #self.comb_forward.setCurrentText('0')
            #self.comb_forward.setCurrentIndex(0)


    def next_img(self):
        file_dir = os.path.join(project_path, self.get_current_file_dir(),"JPEGImages")
        image_id = self.edit_file_name.text()
        file_list = os.listdir(file_dir)
        result_image_id=image_id
        file_index=0
        for file_name in file_list:
            file_index += 1
            file_image_id = file_name[:-4]
            if(file_image_id==image_id):
                if(file_index<len(file_list)):
                    result_image_id=file_list[file_index][:-4]
                break

        self.edit_file_name.setText(result_image_id)
        self.image_info.current_rect_key = None
        #image_id = self.edit_file_name.text()
        #self.edit_file_name.setText(str(int(image_id)+1).zfill(8))
        #self.image_info.current_rect_key = None

        if not self.read_img():
            self.edit_file_name.setText(image_id)
        else:
            self.set_default_image_id()
            self.comb_addkey.setCurrentText('car_idcard')
            self.comb_addkey.setCurrentIndex(0)
            #self.comb_forward.setCurrentText('0')
            #self.comb_forward.setCurrentIndex(0)
    def indent(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for e in elem:
                self.indent(e, level + 1)
            if not e.tail or not e.tail.strip():
                e.tail = i
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
        return elem


    def xml_add_node(self,node,child_tag,child_content):
        ele = ET.Element(child_tag, {})
        if child_content:
            ele.text = child_content
        else:
            ele.text="\n"
        node.append(ele)
        return ele
    def read_img(self):
        file_dir = os.path.join(project_path,self.get_current_file_dir())
        image_id = self.edit_file_name.text()

        file_src=os.path.join( file_dir,"JPEGImages",image_id+'.jpg')
        xml_path = os.path.join(file_dir,"Annotations",image_id+'.xml')
        predict_path =os.path.join(file_dir,"labels_predict",image_id+'.txt')
        rect_img_path = os.path.join(file_dir, "JPEGImages_rect", image_id + ".jpg")
        if not os.path.exists(os.path.dirname(rect_img_path)):
            os.mkdir(os.path.dirname(rect_img_path))
        xml_template_path = os.path.join(file_dir, "template", "0.xml")
        if os.path.exists(file_src):
            if not os.path.exists(xml_path):
                shutil.copy(xml_template_path, xml_path)
            img = cv2.imread(file_src)

            height = img.shape[0]
            width = img.shape[1]
            depth = img.shape[2]
            with open(xml_path) as xml:
                tree = ET.parse(xml)
                root = tree.getroot()
                folder = root.find("folder")
                root.find("filename").text = image_id + '.jpg'
                size = root.find('size')
                size.find('width').text = str(width)
                size.find('height').text = str(height)
                self.label_img.setFixedHeight(height)
                self.label_img.setFixedWidth(width)
                size.find('depth').text = str(depth)
                self.image_info.rects = {}
                objects = root.find("object")
                if not objects or len(objects)==0:
                    if os.path.exists(predict_path):
                        with open(predict_path) as predict_txt:
                            predict = predict_txt.readline()
                            obj_dict = {}
                            while predict and predict!='':
                                predict_array = predict.split(" ")
                                name = predict_array[0]
                                name_array = name.split("_")
                                key = name_array[0]+"_"+name_array[1]
                                forward = name_array[2]
                                if not obj_dict.get(name):
                                    object = self.xml_add_node(root, "object", None)
                                    obj_dict[name] = object
                                    xmin = predict_array[1]
                                    xmax = predict_array[2]
                                    ymin = predict_array[3]
                                    ymax = predict_array[4]
                                    self.xml_add_node(object, "name", name)
                                    self.xml_add_node(object, "key", key)
                                    self.xml_add_node(object, "forward", forward)
                                    self.xml_add_node(object, "pose", 'Unspecified')
                                    self.xml_add_node(object, "truncated", '0')
                                    self.xml_add_node(object, "difficult", '0')
                                    bndbox = self.xml_add_node(object, "bndbox", None)
                                    self.xml_add_node(bndbox, "xmin", str(xmin))
                                    self.xml_add_node(bndbox, "ymin", str(ymin))
                                    self.xml_add_node(bndbox, "xmax", str(xmax))
                                    self.xml_add_node(bndbox, "ymax", str(ymax))
                                    self.indent(root)
                                    #tree.write(xml_path)

                                predict = predict_txt.readline()
                #
                comb_forward_change =False
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    name = obj.find('name').text
                    forward = obj.find("forward").text;
                    if not comb_forward_change:
                        self.comb_forward.setCurrentText(forward)
                        comb_forward_change=True
                    xmin = int(xmlbox.find('xmin').text)
                    ymin = int(xmlbox.find('ymin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymax = int(xmlbox.find('ymax').text)
                    rect = (xmin, ymin, xmax, ymax)
                    self.image_info.rects[name]=rect
                    color = (255, 0, 0)
                    if not self.image_info.current_rect_key:
                        self.image_info.current_rect_key = name
                    if self.image_info.current_rect_key == name:
                        if self.change_mode:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    img = cv2.putText(img, name, (xmax, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imwrite(rect_img_path, img)
                tree.write(xml_path)
            img_read = QtGui.QPixmap(rect_img_path)
            self.label_img.setPixmap(img_read)
            self.statusChange()
            return True
        else:
            QtWidgets.QMessageBox.critical(self, "Critical", "文件不存在:"+file_src)
            print("file error",file_src)
            return  False


    def mousePressEvent(self, event):
        pass

    def keyPressEvent(self, event):
        self.image_info.current_rect_key = None

        file_dir = os.path.join(project_path, self.get_current_file_dir(), "JPEGImages")
        image_id = self.edit_file_name.text()

        file_src = os.path.join(project_path, file_dir, image_id + '.jpg')

        if(not os.path.exists(file_src)):
            self.edit_file_name.setText("");
        self.read_img()
        #print(event)
        pass

    def mouseMoveEvent(self,event):
        #pos = event.pos()
        #print(pos)
        pass


class MyLabel(QtWidgets.QLabel):
    def __init__(self):
        pass

    def __init__(self,main_window):
        super(MyLabel, self).__init__()
        self.main_window = main_window
        self.first_x=0
        self.first_y=0
        self.end_x=0
        self.end_y=0
        self.timer=None

    def timer_function(self):
        #print(self.first_x)
        #print(self.first_y)
        rects = self.main_window.image_info.rects
        #print(rects)
        add = True
        for rect_key in rects:
            rect = rects[rect_key]
            xmin = min(rect[0],rect[2])
            ymin = min(rect[1],rect[3])
            xmax = max(rect[0],rect[2])
            ymax = max(rect[1],rect[3])

            if self.first_x + 5 > xmin and self.first_x - 5<xmax and self.first_y+5>ymin and self.first_y-5<ymax:
                move_x = self.end_x-self.first_x
                move_y = self.end_y-self.first_y
                """"""
                if math.fabs(int(xmin-self.first_x))<3:
                    self.main_window.save_xml(rect_key, "xmin", xmin + move_x)
                elif math.fabs(int(xmax - self.first_x)) < 3:
                    self.main_window.save_xml(rect_key, "xmax", xmax + move_x)
                elif math.fabs(int(ymin - self.first_y)) < 3:
                    self.main_window.save_xml(rect_key, "ymin", ymin + move_y)
                elif math.fabs(int(ymax - self.first_y)) < 3:
                    self.main_window.save_xml(rect_key, "ymax", ymax + move_y)
                else:
                    self.main_window.save_xml(rect_key, "xmin", xmin+move_x, "xmax", xmax+move_x)
                    self.main_window.save_xml(rect_key, "ymin", ymin+move_y, "ymax", ymax+move_y)

                self.main_window.image_info.current_rect_key = rect_key
                add=False
                break
        if add:
            xmin = min(self.first_x, self.end_x)
            ymin = min(self.first_y, self.end_y)
            xmax = max(self.first_x, self.end_x)
            ymax = max(self.first_y, self.end_y)
            file_dir = os.path.join(project_path, self.main_window.get_current_file_dir())
            image_id = self.main_window.edit_file_name.text()
            comb_addkey = self.main_window.comb_addkey.currentText()
            comb_forward = self.main_window.comb_forward.currentText()
            current_name = comb_addkey + '_' + comb_forward
            xml_path = os.path.join(file_dir, "Annotations", image_id + '.xml')
            if os.path.exists(xml_path):
                with open(xml_path) as xml:
                    tree = ET.parse(xml)
                    root = tree.getroot()

                    for obj in root.iter('object'):
                        xmlbox = obj.find('bndbox')
                        name = obj.find('name').text
                        if name == current_name:
                            add = False

                    if add:
                        object = self.main_window.xml_add_node(root, "object", None)
                        self.main_window.xml_add_node(object, "name", current_name)
                        self.main_window.xml_add_node(object, "key", comb_addkey)
                        self.main_window.xml_add_node(object, "forward", comb_forward)
                        self.main_window.xml_add_node(object, "pose", 'Unspecified')
                        self.main_window.xml_add_node(object, "truncated", '0')
                        self.main_window.xml_add_node(object, "difficult", '0')
                        bndbox = self.main_window.xml_add_node(object, "bndbox", None)
                        self.main_window.xml_add_node(bndbox, "xmin", str(xmin))
                        self.main_window.xml_add_node(bndbox, "ymin", str(ymin))
                        self.main_window.xml_add_node(bndbox, "xmax", str(xmax))
                        self.main_window.xml_add_node(bndbox, "ymax", str(ymax))
                        self.main_window.image_info.current_rect_key = current_name
                        self.main_window.indent(root)
                        tree.write(xml_path)
        self.main_window.read_img()
        self.first_x=0
        self.first_y=0
        self.end_x=0
        self.end_y=0






    def mouseMoveEvent(self,event):

        x = event.pos().x()
        y = event.pos().y()
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(0.3,self.timer_function)
        self.timer.start()

        if not self.first_x and not self.first_y:
            self.first_x =x
            self.first_y =y
            self.end_x = x
            self.end_y = y
        else :
            self.end_x = x
            self.end_y = y

    def mousePressEvent(self,event):
        self.mouseMoveEvent(event)

    def keyPressEvent(self, event):
        print(event)
        pass







if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
