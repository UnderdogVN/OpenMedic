import torch
import openmedic.core.shared.services.objects.model as custom_model
from . import utils


class UNet(custom_model.OpenMedicModelBase):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (utils.DoubleConv(n_channels, 64))
        self.down1 = (utils.Down(64, 128))
        self.down2 = (utils.Down(128, 256))
        self.down3 = (utils.Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (utils.Down(512, 1024 // factor))
        self.up1 = (utils.Up(1024, 512 // factor, bilinear))
        self.up2 = (utils.Up(512, 256 // factor, bilinear))
        self.up3 = (utils.Up(256, 128 // factor, bilinear))
        self.up4 = (utils.Up(128, 64, bilinear))
        self.outc = (utils.OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)