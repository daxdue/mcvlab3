import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
from torchvision import transforms
from PIL import Image
import time


print("Insert image name")
imageName = input()


timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("load time {}".format(time.time()-timest))

transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])

img = Image.open(imageName)
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0).cuda()

out = model(batch_t)
print(out.shape)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
