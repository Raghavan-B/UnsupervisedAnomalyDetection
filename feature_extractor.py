from torchvision.models import resnet50,ResNet50_Weights
import torch

class resnet_feature_extractor(torch.nn.Module):
    def __init__(self):
        super(resnet_feature_extractor,self).__init__()
        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        def hook(model,input,output):
            self.features.append(output)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self,input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3,stride = 1)
        fmap_size = self.features[0].shape[-2]
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps,1)
        patch = patch.reshape(patch.shape[1],-1).T

        return patch