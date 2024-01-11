import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

class Hakisa(torch.nn.Module):

    def __init__(self, command_types, actions1, actions2):

        super(Hakisa, self).__init__()

        '''
        Using Convolution Layers with big kernels to try to simulate attention in image regions.
        This may allow for faster convolution without using too many parameters
        Remember that the number of parameters in a conv2d is:
        parameters = input_channels ∙ output_channels ∙ (kerneli ∙ kernelj) ( + bias)

        Y. Xue and J. Qin,
        "Partial Connection Based on Channel Attention for Differentiable Neural Architecture Search,"
        in IEEE Transactions on Industrial Informatics, vol. 19, no. 5, pp. 6804-6813, May 2023, doi: 10.1109/TII.2022.3184700.

        Also keep in mind that Convolution Layers tend to be slow. Avoid using too many, especially at beginning.
        '''

        # 1920 x 1080
        self.conv1 = torch.nn.Conv2d(3, 16, 200, 2, 1, bias=False) # 1,920,000 parameters
        self.conv2 = torch.nn.Conv2d(16, 16, 3, 1, 1, bias=True) # 2,304 parameters + 16 (bias)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, 1, 1, bias=True) # 2,304 parameters + 16 (bias)
        # 862 x 442
        self.conv4 = torch.nn.Conv2d(16, 32, 100, 2, 1, bias=False) # 5,120,000 parameters
        self.conv5 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True) # 9,216 parameters + 32
        self.conv6 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True) # 9,216 parameters + 32
        # 383 x 173
        self.conv7 = torch.nn.Conv2d(32, 64, 50, 2, 0, bias=False) # 5,120,000 parameters
        self.conv8 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True) # 36,864 parameters
        self.conv9 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True) # 36,864 parameters
        # 167 x 62
        self.conv10 = torch.nn.Conv2d(64, 128, 25, 2, 0, bias=False) # 5,120,000 parameters
        self.conv11 = torch.nn.Conv2d(128, 128, 3, 1, 1, bias=True) # 147,456 parameters
        self.conv12 = torch.nn.Conv2d(128, 128, 3, 1, 1, bias=True) # 147,456 parameters
        # 72 x 19
        self.neuron_in = torch.nn.Linear(72*19*128, 256, bias=True) # Bottleneck layer = 44,826,624 parameters

        self.preluA = torch.nn.PReLU()
        self.preluB = torch.nn.PReLU()
        self.preluC = torch.nn.PReLU()
        self.preluD = torch.nn.PReLU()
        self.preluE = torch.nn.PReLU()
        self.preluF = torch.nn.PReLU()
        self.preluG = torch.nn.PReLU()
        self.preluH = torch.nn.PReLU()

        self.pred_command_type = torch.nn.Linear(256, len(command_types))
        self.pred_action1 = torch.nn.Linear(256, len(actions1))
        self.pred_action2 = torch.nn.Linear(256, len(actions2))

    def forward(self, input):
        # To condition decision based on output --> Embedding layer to input image

        x = self.conv1(input)
        x = self.preluA(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.preluB(x)

        del input

        x = self.conv4(x)
        x = self.preluC(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.preluD(x)

        x = self.conv7(x)
        x = self.preluE(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.preluF(x)

        x = self.conv10(x)
        x = self.preluG(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.preluH(x)

        x = x.view(x.size(0), -1)

        x = self.neuron_in(x)

        command_type = self.pred_command_type(x)
        action1 = self.pred_action1(x)
        action2 = self.pred_action2(x)

        return (command_type, action1, action2)