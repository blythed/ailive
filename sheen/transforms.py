import torchvision.transforms


def get_augment_transform(image_size=160):
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
