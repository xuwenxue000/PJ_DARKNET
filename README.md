# 说明
    给服务传入一个发票图片文件的路径,就可以返回对这个图片预测出来的身份证,vin码,发动机号,价格所在的区域的坐标,
    目前支持的是方形的图片,图片翻转没关系,可以识别出来,
    比如正向的是car_idcard_0,右转90度的就是:car_idcard_1,右转180的是car_idcard_2,右转270的是car_idcard_3
    返回的坐标以及分类可以准确的切出对应区域的图片并将其旋转为正确的方向
# 使用示例
- 下载
    - git clone https://github.com/xuwenxue000/PJ_DARKNET.git
- 环境要求
    - 如果走gpu
        - cuda
        - cnn
    - python3
    
- 编译
    - 如果使用GPU跑,需要改下makefile
        - GPU=1
        - CUDNN=1
    - opencv的由报错.暂时没处理,不要打开
    - 配置好makfile,执行命令,生成darknet命令
        - make    
- 解压模型
    - 模型目录:backup
    - 因为github的大小限制,模型有点大,所以使用了压缩工具压缩分割了.下载完成之后,需要再使用压缩工具镜像解压,
    - 解压后文件名应为:yolo-invoice_final.weight
- 启动服务
    - ./darknet
    
- 访问
    - http://localhost:8088/invoice?/Users/william/PycharmProjects/darknet_cp/00000001
    - 会返回对应的json数据,标识每个识别出来的区域的坐标
 
# 其他:
- markutil.py是用来做图片的标识的工具
- invoice_label.py 生成训练所许文件
- 训练命令:
    ./darknet detector train cfg/invoice.data cfg/yolo-invoice.cfg darknet19_448.conv.23 -gpus 0,1
- 批量检测命令:
    - ./darknet detector test_batch cfg/invoice.data cfg/yolo-invoice.cfg backup/yolo-invoice.backup predict_list.txt
    - predict_list.txt里面列出了所有要识别的文件列表
- 源darknet网址
    https://pjreddie.com/darknet/
        
