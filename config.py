import argparse
import os
import sys
root = os.getcwd()
class Config(object):
    params = {
        "root": root,
        "mode": "default",
        "keyword": "default",
        "epoch": 10 ,
        "learning_rate": 0.03,
        "batch_size": 16,
        "weight_decay": 0.0001,
        "driver_path": root + r"\chromedriver_win32\chromedriver.exe",
        "image_save_path": root + r"\result",
        "model_load_path": root + r"\save_model\06M 15D 01 30M 3E.tar",
        "model_save_path": root + r'\save_model',
        "image_folder": r"G:\기계학습프로젝트데이터",
        "test_image": r"G:\기계학습프로젝트데이터\vad\독버섯\Amanita melleiceps_img_79.jpg.jpg",
        "top3": [],
        "class_num": 2,
        "argumentation": False
    }
# tensor([[4.3608e-04, 9.9956e-01]], device='cuda:0', grad_fn=<SoftmaxBackward>)
