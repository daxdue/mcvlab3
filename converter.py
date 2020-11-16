from models import *
from utils import *

import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

x1 = torch.ones((1, 3, 224, 224)).cuda()


config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

model_trt = torch2trt(model, [x1], use_onnx=True)
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
exit()
