import torchvision.transforms


def get_augment_transform(imageSize=160):
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(imageSize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
