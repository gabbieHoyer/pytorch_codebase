'''
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def initialize_model(model_name, num_classes, use_as_feature_extractor, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        if use_as_feature_extractor:
            set_parameter_requires_grad(model_ft, use_as_feature_extractor)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.8),
                                    nn.Linear(512, num_classes))
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        if use_as_feature_extractor:
            set_parameter_requires_grad(model_ft, use_as_feature_extractor)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        if use_as_feature_extractor:
            set_parameter_requires_grad(model_ft, use_as_feature_extractor)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        if use_as_feature_extractor:
            set_parameter_requires_grad(model_ft, use_as_feature_extractor)
        model_ft.classifier[1] = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1)),
                                            nn.Dropout(p=0.8),
                                            nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        if use_as_feature_extractor:
            set_parameter_requires_grad(model_ft, use_as_feature_extractor)
        num_ftrs = model_ft.classifier.in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.8),
                                    nn.Linear(512, num_classes))

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft
