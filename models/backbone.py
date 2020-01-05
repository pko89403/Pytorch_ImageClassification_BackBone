import os
import torch
import torchvision.models as models
from torchvision.models import *
from torch import nn
from torch import optim
import torch.nn.functional as F
from config import config

class bb_fc_model(object):
    def __init__(self, model_name, pretrained=True):
        
        self.out_feature_num = config().json_data["MODEL"]["CLASS"]
        self.learning_rate = config().json_data["MODEL"]["LEARNING_RATE"]
        self.epoch = config().json_data["MODEL"]["EPOCH"]
        self.print_step = config().json_data["MODEL"]["PRINT_STEP"]
        
        self.model = None
        self.in_features = 0
        print( model_name )

        if model_name == "resnet18":
            self.model = resnet18(pretrained=pretrained, progress=False)
            self.in_features = 512
        elif model_name == "alexnet":
            self.model = alexnet(pretrained=pretrained, progress=False)
            self.in_features = 1000
        elif model_name == 'vgg16':
            self.model = vgg16(pretrained=pretrained, progress=False)
            self.in_features = 1000
        elif model_name == 'squeezenet':
            self.model = squeezenet1_0(pretrained=pretrained, progress=False)
            self.in_features = 1000,
        elif model_name == 'densenet':
            self.model = densenet161(pretrained=pretrained, progress=False)
            self.in_features = 1000
        elif model_name == 'inception':
            self.model = inception_v3(pretrained=pretrained, progress=False)
            self.in_features = 1024
        elif model_name == 'googlenet':
            self.model = googlenet(pretrained=pretrained, progress=False)
            self.in_features = 512
        elif model_name == 'shufflenet':
            self.model = shufflenet_v2_x1_0(pretrained=pretrained, progress=False)
            self.in_features = 512
        elif model_name == 'mobilenet':
            self.model = mobilenet_v2(pretrained=pretrained, progress=False)
            self.in_features = 1000
        elif model_name == 'resnext50_32x4d':
            self.model = resnext50_32x4d(pretrained=pretrained, progress=False)
            self.in_features = 2048
        elif model_name == 'wide_resnet50_2':
            self.model = wide_resnet50_2(pretrained=pretrained, progress=False)
            self.in_features = 2048
        elif model_name == 'mnasnet':
            self.model = mnasnet1_0(pretrained=pretrained, progress=False)
            self.in_features = 1000
        else:
            pass
        self.model_name = model_name
        # freeze pre-trained layers ( don't do backprop during training )
        for param in self.model.parameters():
            param.requires_grad = False


        self.model.fc = nn.Sequential(
                        nn.Linear(in_features=self.in_features, out_features=512, bias=True),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(in_features=512, out_features=self.out_feature_num),
                        nn.LogSoftmax(dim=1)
                        )
        # negative log likelihood loss. it is useful to train classification problem with C classes
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        # Moves and/or casts the parameters and buffers
        # only accepts floating point desired dtype s.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        print(f"cuda is available : {torch.cuda.is_available()}")
        
        self.model = nn.DataParallel(self.model)
        self.model.to(device=self.device)
        print(self.model)
        
    def train(self, train_loader, valid_loader):
        optimizer = self.optimizer
        loss = self.criterion
        model = self.model
        device = self.device
        epochs = self.epoch

        steps = 0
        running_loss = 0
        print_every = 10
        train_losses, valid_losses = [], []
        train_loader = train_loader

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                losses = loss(logps, labels)
                losses.backward()
                optimizer.step()
                running_loss += losses.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = loss(logps, labels)
                            valid_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    train_losses.append(running_loss / len(train_loader))
                    valid_losses.append(valid_loss / len(valid_loader))
                    print(f"Epoch  {epoch + 1} / { epochs } ..")
                    print(f"Train loss : {running_loss / print_every:.3f} ..")
                    print(f"Valid loss : {valid_loss / len(valid_loader):.3f} ..")
                    print(f"Valid accuracy : {accuracy / len(valid_loader):.3f} ..")
                    running_loss = 0
                    model.train()
        
    def save_model(self, file_path=None):
        name = self.model_name + '.pth'
        save_file_name = os.path.join(file_path, name)
        torch.save(obj=self.model, f=save_file_name)

        


