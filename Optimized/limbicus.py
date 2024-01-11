import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

'''
The Limbic System is actually a group of different organs with their specialized regions

It was previously thought to be simply the brain limbic lobus, then it was thought to be a complex
circuit, the Papez Circuit.

Nowadays, it's classified as a complex system, which is composed of brain's limbic lobus,
brain's hypocampus and basal nuclei, brain's olfatory cortex, hypothalamus and thalamus and mesencephalus

Hypothalamus participation is the main responsible for physiological responses to emotions, like fear and happiness.
The amygdala does the same, but since it's a brain nucleus, it tends to also work with more conscient processes, while helping the hypothalamus.

Hypocampus is specialized in work memory and short-term memory, and it's the entrance door for long-term memory in the brain

The shining star may probably be the Nucleus Accumbens, which is modulated by dopamine originated from the Ventral Tegmental Area and
Substantia Nigra in the mesencephalus (midbrain). The dopamine makes the neurons in Nucleus Accumbens easier to be excited and depolarized(activated),
which originates not only reward, but also motivation, since the reward expectancy motivates the subject to make a move.
Sometimes, the pleasure from expectation can be higher than the action reward stimulus
(The expectation of getting a delicious food may provide more pleasure than actually eating the food)

MENESES, M. Neuroanatomia Aplicada, 3 ed. Guanabara Koogan.
https://en.wikipedia.org/wiki/Nucleus_accumbens
https://en.wikipedia.org/wiki/Ventral_tegmental_area
RICE, M. Closing in on what motivates motivation, https://doi.org/10.1038/d41586-019-01589-6
'''

class ResidualBlock(torch.nn.Module):

    def __init__(self, input_channels, kernel_size, strides=1, padding=1):

        super(ResidualBlock, self).__init__()

        self.convA = torch.nn.Conv2d(input_channels, input_channels, kernel_size, strides, padding, bias=True)
        self.batchnormA = torch.nn.BatchNorm2d(input_channels)
        self.convB = torch.nn.Conv2d(input_channels, input_channels, kernel_size, strides, padding, bias=True)
        self.batchnormB = torch.nn.BatchNorm2d(input_channels)

        self.PRelu = torch.nn.PReLU()

    def forward(self, input):

        x = self.convA(input)
        x = self.batchnormA(x)
        x = self.PRelu(x)
        x = self.convB(x)
        x = self.batchnormB(x)

        output = input + x

        del x

        return output
    
class LimbicSystemA(torch.nn.Module):
    '''
    Reward model, based on ResNet architecture
    '''

    def __init__(self):

        super(LimbicSystemA, self).__init__()

        self.conv_in = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        
        self.resblock1 = ResidualBlock(64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.resblock4 = ResidualBlock(128, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv6 = torch.nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.resblock7 = ResidualBlock(256, 3, 1, 1)

        self.batch_norm = torch.nn.BatchNorm1d(3072)

        self.neuronA = torch.nn.Linear(3072, 1024, bias=True)
        self.neuronB = torch.nn.Linear(1024, 128, bias=True)
        self.neuronC = torch.nn.Linear(128, 16, bias=True)

        self.neuron_out = torch.nn.Linear(16, 1, bias=True)

        self.pool = torch.nn.AvgPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)

        self.LRelu = torch.nn.LeakyReLU(0.25)

    def forward(self, input):

        x = self.conv_in(input)
        x = self.LRelu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.resblock1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.LRelu(x)
        #x = self.dropout(x)
        
        x = self.resblock4(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.LRelu(x)
        #x = self.dropout(x)

        x = self.resblock7(x)

        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.batch_norm(x)
        x = self.neuronA(x)
        x = self.LRelu(x)
        x = self.neuronB(x)
        x = self.LRelu(x)
        x = self.neuronC(x)
        x = self.LRelu(x)
    
        output = self.neuron_out(x)

        return output
    
class LimbicSystemB(torch.nn.Module):
    '''
    Reward model, based on ResNet architecture
    '''

    def __init__(self):

        super(LimbicSystemB, self).__init__()

        self.conv_in = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        
        self.resblock1 = ResidualBlock(64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.resblock4 = ResidualBlock(128, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv6 = torch.nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.resblock7 = ResidualBlock(256, 3, 1, 1)
        self.conv8 = torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv9 = torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.resblock10 = ResidualBlock(512, 3, 1, 1)

        self.batch_norm = torch.nn.BatchNorm1d(12288)

        self.neuronA = torch.nn.Linear(12288, 512, bias=True)
        self.neuronB = torch.nn.Linear(512, 128, bias=True)
        self.neuronC = torch.nn.Linear(128, 16, bias=True)

        self.neuron_out = torch.nn.Linear(16, 1, bias=True)

        self.pool = torch.nn.AvgPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)

        self.LRelu = torch.nn.LeakyReLU(0.25)

    def forward(self, input):

        x = self.conv_in(input)
        x = self.LRelu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.resblock1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.LRelu(x)
        #x = self.dropout(x)
        
        x = self.resblock4(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.LRelu(x)
        #x = self.dropout(x)

        x = self.resblock7(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.LRelu(x)
        #x = self.dropout(x)

        x = self.resblock10(x)

        x = self.pool(x) # 4x6
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.batch_norm(x)
        x = self.neuronA(x)
        x = self.LRelu(x)
        x = self.neuronB(x)
        x = self.LRelu(x)
        x = self.neuronC(x)
        x = self.LRelu(x)
    
        output = self.neuron_out(x)

        return output
    
class LimbicSystemC(torch.nn.Module):
    '''
    Reward model, based on ResNet architecture
    '''

    def __init__(self):

        super(LimbicSystemC, self).__init__()

        self.conv_in = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        
        self.resblock1 = ResidualBlock(64, 3, 1, 1)
        self.resblock2 = ResidualBlock(64, 3, 1, 1)
        self.resblock3 = ResidualBlock(64, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv5 = torch.nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.resblock6 = ResidualBlock(128, 3, 1, 1)
        self.resblock7 = ResidualBlock(128, 3, 1, 1)
        self.resblock8 = ResidualBlock(128, 3, 1, 1)
        self.conv9 = torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv10 = torch.nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.resblock11 = ResidualBlock(256, 3, 1, 1)
        self.resblock12 = ResidualBlock(256, 3, 1, 1)
        self.resblock13 = ResidualBlock(256, 3, 1, 1)
        self.conv14 = torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv15 = torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.resblock16 = ResidualBlock(512, 3, 1, 1)
        self.resblock17 = ResidualBlock(512, 3, 1, 1)
        self.resblock18 = ResidualBlock(512, 3, 1, 1)

        self.batch_norm = torch.nn.BatchNorm1d(36864)

        self.neuronA = torch.nn.Linear(36864, 256, bias=True)
        self.neuronB = torch.nn.Linear(256, 256, bias=True)
        self.neuronC = torch.nn.Linear(256, 64, bias=True)

        self.neuron_out = torch.nn.Linear(64, 1, bias=True)

        self.pool = torch.nn.AvgPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)

        self.LRelu = torch.nn.LeakyReLU(0.25)

    def forward(self, input):

        x = self.conv_in(input)
        x = self.LRelu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.LRelu(x)
        #x = self.dropout(x)
        
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.LRelu(x)
        #x = self.dropout(x)

        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)

        x = self.conv14(x)
        x = self.conv15(x)
        x = self.LRelu(x)
        #x = self.dropout(x)

        x = self.resblock16(x)
        x = self.resblock17(x)
        x = self.resblock18(x)

        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.batch_norm(x)
        x = self.neuronA(x)
        x = self.LRelu(x)
        x = self.neuronB(x)
        x = self.LRelu(x)
        x = self.neuronC(x)
        x = self.LRelu(x)
    
        output = self.neuron_out(x)

        return output