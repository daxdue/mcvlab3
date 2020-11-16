import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time

x1 = torch.ones((1, 3, 224, 224)).cuda()

model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

model_trt = torch2trt(model, [x1], use_onnx=True)
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
exit()
