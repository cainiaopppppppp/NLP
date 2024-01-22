import os
import requests
import zipfile
from tqdm import tqdm
from shutil import copyfile
def move(save_url,file_name,path_test,path_train):
    zip_train=os.path.join(save_url,'original/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt')
    zip_test=os.path.join(save_url,'original/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt')
    with zipfile.ZipFile(os.path.join(save_url, file_name)) as zf:
        for name in zf.namelist():
            zf.extract(name,path=save_url)
    copyfile(zip_train,path_train)
    copyfile(zip_test,path_test)
def Download(data_url, save_url,file_name):
    folder = os.path.exists(save_url)
    if not folder:
        os.makedirs(save_url)
    # 读取data资源
    res = requests.get(data_url,stream=True) 
    total_size = int(int(res.headers["Content-Length"])/1024+0.5)
    # 获取文件地址
    file_path = os.path.join(save_url, file_name)
    
    # 打开本地文件夹路径file_path，以二进制流方式写入，保存到本地
    with open(file_path, 'wb') as fd:
        print('开始下载文件：{},当前文件大小：{}KB'.format(file_name,total_size))
        for chunk in tqdm(iterable=res.iter_content(1024),total=total_size,unit='k',desc=None):
            fd.write(chunk)
        print(file_name+' 下载完成！')
   
