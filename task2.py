import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as utils
import cv2
import numpy as np
from PIL import Image
import argparse

"""
input commands
"""
paser = argparse.ArgumentParser()
paser.add_argument("--test_img", type=str, default='whippet.jpg', help="testing image")
opt = paser.parse_args()


# main
if __name__ == "__main__":
    """
    data transforms, for pre-processing the input testing image before feeding into the net
    """
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

    # obtain the file path of the testing image
    test_image_dir = './alexnet_images'
    test_image_filepath = os.path.join(test_image_dir, opt.test_img)
    #print(test_image_filepath)

    # open the testing image
    img = Image.open(opt.test_img)
    print("original image's shape: " + str(img.size))
    # pre-process the input
    transformed_img = data_transforms(img)
    print("transformed image's shape: " + str(transformed_img.shape))
    # form a batch with only one image
    batch_img = torch.unsqueeze(transformed_img, 0).cuda()
    print("image batch's shape: " + str(batch_img.shape))

    # load pre-trained AlexNet model
    print("\nfeed the input into the pre-trained alexnet to get the output")
    alexnet = models.alexnet(pretrained=True)
    # put the model to eval mode for testing
    alexnet.eval().cuda()

    # obtain the output of the model
    output = alexnet(batch_img).cuda()
    print("output vector's shape: " + str(output.shape))

    # obtain the activation maps
    #visualize_activation_maps(batch_img, alexnet)

    # map the class no. to the corresponding label
    with open('class_names_ImageNet.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]

    # print the first 5 classes to see the labels
    print("\nprint the first 5 classes to see the lables")
    for i in range(5):
        print("class " + str(i) + ": " + str(classes[i]))

    # sort the probability vector in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = F.softmax(output, dim=1)[0] * 100.0
    # obtain the first 5 classes (with the highest probability) the input belongs to
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
    print("\nprint the first 5 classes the testing image belongs to")
    for i in range(5):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))
