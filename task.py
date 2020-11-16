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

out = model(batch_t).cuda()

with open('class_names_ImageNet.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]

    # print the first 5 classes to see the labels

    for i in range(5):
        print("class " + str(i) + ": " + str(classes[i]))

    # sort the probability vector in descending order
    sorted, indices = torch.sort(out, descending=True)
    percentage = F.softmax(out, dim=1)[0] * 100.0
    # obtain the first 5 classes (with the highest probability) the input belongs to
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
    print("\nprint the first 5 classes the testing image belongs to")
    for i in range(5):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))
