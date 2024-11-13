from torchvision import transforms

def img_transformer():
    ## transformation
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform