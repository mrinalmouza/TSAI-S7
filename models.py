import torch.nn as nn
import torch.nn.functional as F

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        #Input shape [512, 1, 28, 28]

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3),
                                        nn.ReLU())
        #Output shape of above block  [512, 1, 26, 26] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 32, out_channels = 64, kernel_size= 3),
                                        nn.ReLU())
        
        #Output shape of above block  [512, 1, 24, 24] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)

        ##Output shape of above block  [512, 1, 12, 12] , RF >> 6

        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3),
                                        nn.ReLU())
        
        #Output shape of above block  [512, 1, 10, 10] , RF >> 10

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels= 256, kernel_size= 3),
                                        nn.ReLU())

        #Output shape of above block  [512, 1, 8, 8] , RF >> 14

        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 4, 4] , RF >> 15

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels= 512, kernel_size= 3),
                                                  nn.ReLU())
        
        #Output shape of above block  [512, 1, 2, 2] , RF >> 23

        self.conv6 = nn.Conv2d(in_channels= 512, out_channels= 10, kernel_size= 2)
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        #Output shape of above block  [512, 1, 1, 1] , RF >> 31

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.conv6(x)
        #x = self.avg_pool(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Build a basic model that can predict correctly the digits. The emphasis is on complete execution.
    # Result: The model trained for 15 epocs, it reached maximum of  99.90 in 11th epoch of training cycle and  accuracy 9.31 in 13th epoch for  test.
    # Analysis: The model is overfitting, also the model has 1.588 Million parameters which is way too much for this problem set.

class model2(nn.Module):
    def __init__(self) -> None:
        super(model2, self).__init__()
        #Input shape [512, 1, 28, 28]
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding=1),
                                        nn.ReLU())
        #Output shape of above block  [512, 1, 28, 28] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 64, out_channels = 128, kernel_size= 3, padding= 1),
                                        nn.ReLU())
        
        #Output shape of above block  [512, 1, 28, 28] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)
        #Output shape of above block [512, 1, 14, 14] , RF >> 6
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 128, out_channels= 16, kernel_size= 1),
                                        nn.ReLU())
        ##Output shape of above block  [512, 1, 14, 14] , RF >> 6

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels= 16, out_channels= 64, kernel_size= 3, padding=1),
                                        nn.ReLU())
        
        #Output shape of above block  [512, 1, 14, 14] , RF >> 10
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 7, 7] , RF >> 12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels= 16, kernel_size= 3),
                                        nn.ReLU())

        #Output shape of above block  [512, 1, 5, 5] , RF >> 20

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= 3))
        #Output shape of above block  [512, 1, 3, 3] , RF >> 28

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                                                  
        #Output shape of above block  [512, 1, 1, 1] , RF >> 28

        #self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels= 10, kernel_size= 2))
        #Output shape of above block  [512, 1, 1, 1] , RF >> 21

        #self.conv6 = nn.Conv2d(in_channels= 512, out_channels= 10, kernel_size= 2)

        #Output shape of above block  [512, 1, 1, 1] , RF >> 28

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Reduce the number of parameters and keep the train and test accuracy high. Make a skeleton that is good.
    # Result: Made a model with 96522. The highest training accuracy reached was 99.88 and testing was 99.25.
    # Analysis: The model is good but still has lot of parametes. Also the model overfitting. But the skeleton is good.


class model3(nn.Module):
    def __init__(self) -> None:
        super(model3, self).__init__()
        #Input shape [512, 1, 28, 28]
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 40, kernel_size = 3, padding=1),
                                        nn.ReLU())
        #Output shape of above block  [512, 1, 28, 28] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 40, out_channels = 70, kernel_size= 3, padding= 1),
                                        nn.ReLU())
        
        #Output shape of above block  [512, 1, 28, 28] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)
        #Output shape of above block [512, 1, 14, 14] , RF >> 6
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 70, out_channels= 16, kernel_size= 1),
                                        nn.ReLU())
        ##Output shape of above block  [512, 1, 14, 14] , RF >> 6

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels= 16, out_channels= 64, kernel_size= 3, padding=1),
                                        nn.ReLU())
        
        #Output shape of above block  [512, 1, 14, 14] , RF >> 10
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 7, 7] , RF >> 12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels= 16, kernel_size= 3),
                                        nn.ReLU())

        #Output shape of above block  [512, 1, 5, 5] , RF >> 20

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= 3))
        #Output shape of above block  [512, 1, 3, 3] , RF >> 28

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                                                  
        #Output shape of above block  [512, 1, 1, 1] , RF >> 28

        #self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels= 10, kernel_size= 2))
        #Output shape of above block  [512, 1, 1, 1] , RF >> 21

        #self.conv6 = nn.Conv2d(in_channels= 512, out_channels= 10, kernel_size= 2)

        #Output shape of above block  [512, 1, 1, 1] , RF >> 28

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Reduce the number of parameters and keep the train and test accuracy high. Make a skeleton that is good.
    # Result: Made a model with 46768. The highest training accuracy reached was 99.70 and testing was 99.19.
    # Analysis: The model is good but still has lot of parametes. Also the model overfitting. The training and testing accuracy
    # has lot of gap. We need to introduce batch normalization and also reduce number of parameters.
    
class model4(nn.Module):
    def __init__(self) -> None:
        super(model4, self).__init__()
        #Input shape [512, 1, 28, 28]
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(10))
        #Output shape of above block  [512, 1, 28, 28] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 10, out_channels = 20, kernel_size= 3, padding= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(20))
        
        #Output shape of above block  [512, 1, 28, 28] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)
        #Output shape of above block [512, 1, 14, 14] , RF >> 6
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 20, out_channels= 10, kernel_size= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(10))
        ##Output shape of above block  [512, 1, 14, 14] , RF >> 6

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels= 10, out_channels= 20, kernel_size= 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(20))
        
        #Output shape of above block  [512, 1, 14, 14] , RF >> 10
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 7, 7] , RF >> 12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels= 16, kernel_size= 3, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16))

        #Output shape of above block  [512, 1, 5, 5] , RF >> 20

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= 3))
        #Output shape of above block  [512, 1, 3, 3] , RF >> 28

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                                                  
        #Output shape of above block  [512, 1, 1, 1] , RF >> 28


    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Reduce the number of parameters below 9K and keep the train and test accuracy high.
    # Result: Made a model with 8372 parameters. The highest training accuracy reached was 99.89 and testing was 99.21.
    # Analysis: The model is good. Even after adding batch normalization the model is overfitting. 
    # The training and testing accuracy still has lot of gap. 
    # We need to introduce more regularization by introducing dropout
    
class model5(nn.Module):
    def __init__(self) -> None:
        super(model5, self).__init__()
        #Input shape [512, 1, 28, 28]
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(8),
                                        nn.Dropout2d(.1))
        #Output shape of above block  [512, 1, 28, 28] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size= 3, padding= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.1))
        
        #Output shape of above block  [512, 1, 28, 28] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)
        #Output shape of above block [512, 1, 14, 14] , RF >> 6
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(10),
                                        nn.Dropout2d(.1))
        ##Output shape of above block  [512, 1, 14, 14] , RF >> 6

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels= 10, out_channels= 20, kernel_size= 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(20),
                                        nn.Dropout2d(.1))
        
        #Output shape of above block  [512, 1, 14, 14] , RF >> 10
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 7, 7] , RF >> 12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels= 16, kernel_size= 3, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.1))

        #Output shape of above block  [512, 1, 5, 5] , RF >> 20

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= 3))
        #Output shape of above block  [512, 1, 3, 3] , RF >> 28

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                                                  
        #Output shape of above block  [512, 1, 1, 1] , RF >> 28


    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Reduce the number of parameters below 8K and keep the train and test accuracy high. Remove overfitting.
    # Result: Made a model with 7654 parameters. The highest training accuracy reached was 99.05 and testing was 99.30
    # Analysis: The model is good. It is not overfitting. There is a little bit of underfit which can be finetuned further
    # to bring the testing accuracy a little higher. Adding of dropout worked as it made the training harder which reduced the 
    # training accuracy a bit but kept the testing accuracy high.

class model6(nn.Module):
    def __init__(self) -> None:
        super(model6, self).__init__()
        #Input shape [512, 1, 28, 28]
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(8),
                                        nn.Dropout2d(.02))
        #Output shape of above block  [512, 1, 28, 28] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size= 3, padding= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.02))
        
        #Output shape of above block  [512, 1, 28, 28] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)
        #Output shape of above block [512, 1, 14, 14] , RF >> 6
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(10),
                                        nn.Dropout2d(.02))
        ##Output shape of above block  [512, 1, 14, 14] , RF >> 6

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels= 10, out_channels= 20, kernel_size= 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(20),
                                        nn.Dropout2d(.02))
        
        #Output shape of above block  [512, 1, 14, 14] , RF >> 10
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 7, 7] , RF >> 12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels= 16, kernel_size= 3, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.02))

        #Output shape of above block  [512, 1, 5, 5] , RF >> 20

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= 3))
        #Output shape of above block  [512, 1, 3, 3] , RF >> 28

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                                                  
        #Output shape of above block  [512, 1, 1, 1] , RF >> 28


    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Improve underfitting of the model by fine tuning the dropout value. Also introduce data augmentation techniques
    # Result: Made a model with 7654 parameters. The highest training accuracy reached was 99.68 and testing was 99.41
    # Analysis: The model is good. It is neither overfitting or underfitting. 
    # The dropout value of 0.02 seems to be the sweet spot
    
class model7(nn.Module):
    def __init__(self) -> None:
        super(model7, self).__init__()
        #Input shape [512, 1, 28, 28]
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(8),
                                        nn.Dropout2d(.02))
        #Output shape of above block  [512, 1, 28, 28] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size= 3, padding= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.02))
        
        #Output shape of above block  [512, 1, 28, 28] , RF >> 5

        self.maxpool1 = nn.MaxPool2d(2, 2)
        #Output shape of above block [512, 1, 14, 14] , RF >> 6
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= 1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(10),
                                        nn.Dropout2d(.02))
        ##Output shape of above block  [512, 1, 14, 14] , RF >> 6

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels= 10, out_channels= 20, kernel_size= 3, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(20),
                                        nn.Dropout2d(.02))
        
        #Output shape of above block  [512, 1, 14, 14] , RF >> 10
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #Output shape of above block  [512, 1, 7, 7] , RF >> 12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels= 16, kernel_size= 3, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.02))

        #Output shape of above block  [512, 1, 5, 5] , RF >> 20

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= 3))
        #Output shape of above block  [512, 1, 3, 3] , RF >> 28

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                                                  
        #Output shape of above block  [512, 1, 1, 1] , RF >> 28


    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = 1)

        return x

    # Target: Consistently hit testing accuracy of 99.40.  Introduce data augmentation techniques to improve the accuracy.
    # Result: Made a model with 7654 parameters. The highest training accuracy reached was 99.11 and testing was 99.44
    # Analysis: Introduction of image rotation by.15 improve the testing accuracy.
    # The model hit consistently hit testing accuracy of greater than 99.40 for 4 epochs. Also i was able to replicate
    # the results by running multiple times.
    # MISSION ACCOMPLISHED!!!! Hit 99.4 consitently with less thank 8k parmeters. WOhoo
    
