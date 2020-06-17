import torchvision
import torch
import os
import PIL
import matplotlib.pyplot as plt

from utils.config import Config

# load images and set label
def load_data(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    argumentation_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),            # 색상 변환
        torchvision.transforms.RandomVerticalFlip(),  # 세로 뒤집기
        torchvision.transforms.RandomHorizontalFlip(),  # 가로 뒤집기
        #torchvision.transforms.RandomCrop(180, padding=4),                      # 아무곳이나 자르기
        #torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR), # 회전 시키기
        torchvision.transforms.ToTensor()
    ])

    train_dir = os.path.join(config["image_folder"], "train")
    vad_dir = os.path.join(config["image_folder"], "vad")

    # config에서 argumentation True로 해두시면 설정될거에요
    if config['argumentation']:
        transform = argumentation_transform

    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

    vad_set = torchvision.datasets.ImageFolder(root=vad_dir, transform=transform)
    vad_loader = torch.utils.data.DataLoader(vad_set, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    ## 이미지 argumentaion 되는거 보고싶으시면 주석 풀면 되십니다
    ## 일단 주석처리 해둘게요
    # for img, _ in train_loader:
    #     plt.imshow(img[0].permute(1, 2, 0).numpy().squeeze())
    #     plt.show()

    if len(train_set.classes) == len(vad_set.classes):
        config["class_num"] = len(train_set.classes)
    else:
        print("set train and test classes to be same")
    return train_loader, vad_loader
