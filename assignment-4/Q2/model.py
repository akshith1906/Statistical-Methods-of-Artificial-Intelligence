import torch
import torch.nn as nn

class ColorNet(nn.Module):
    """
    Encoder-Decoder CNN for Image Colorization based on SMAI A4 Q2.
    """
    def __init__(self, NIC=1, NF=32, NC=24, kernel_size=3):
        super(ColorNet, self).__init__()
        
        padding = (kernel_size - 1) // 2 

        # Encoder (Down-sampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(NIC, NF, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(NF),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(NF, 2*NF, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(2*NF),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(2*NF, 4*NF, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(4*NF),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        )
        
        # Decoder (Up-sampling path) 
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(4*NF, 2*NF, kernel_size=2, stride=2), # 4x4 -> 8x8
            nn.BatchNorm2d(2*NF),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(2*NF, NF, kernel_size=2, stride=2), # 8x8 -> 16x16
            nn.BatchNorm2d(NF),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(NF, NC, kernel_size=2, stride=2), # 16x16 -> 32x32
            nn.BatchNorm2d(NC),
            nn.ReLU(inplace=True)
        )

        # Final 1x1 Convolution (Classifier)
        self.classifier = nn.Conv2d(NC, NC, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Decoder
        x4 = self.dec1(x3)
        x5 = self.dec2(x4)
        x6 = self.dec3(x5)
        
        # Final Classification
        out = self.classifier(x6) # Shape [B, NC, 32, 32]
        return out

# This block lets you test the file directly ---
if __name__ == "__main__":
    print("Testing model.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test with NF=32
    model = ColorNet(NIC=1, NF=32, NC=24, kernel_size=3).to(device)
    
    # Create a dummy batch
    dummy_input = torch.randn(64, 1, 32, 32).to(device)
    
    # Pass through model
    output_logits = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    
    if output_logits.shape == (64, 24, 32, 32):
        print("Model shape test PASSED.")
    else:
        print("Model shape test FAILED.")