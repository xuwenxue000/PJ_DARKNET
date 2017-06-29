# 使用示例
- 编译
    - 如果使用GPU跑,需要改下makefile
    - opencv的由报错.暂时没处理,不要打开
    - 配置好makfile,执行make
    - 生成darknet命令
- 解压模型
    - 模型目录:backup
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
        
