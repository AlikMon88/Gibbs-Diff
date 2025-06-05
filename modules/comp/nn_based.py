import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenoisingAutoencoder(nn.Module):
    
    def __init__(self, input_shape = (100, 1)):
        super().__init__()
        self.input_shape = input_shape

        ## Conv + MaxPooling
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = self.input_shape[-1], out_channels = 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels = 16, out_channels = 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size = 2, stride = 2)
            
        )
        
        ## DeConv + UpSampling
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode = 'nearest'), ### Upsampling = Interpolation with Nearest Neighbours 
            nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode = 'nearest'), ### Upsampling = Interpolation with Nearest Neighbours 
            nn.Conv1d(in_channels = 16, out_channels = self.input_shape[-1], kernel_size = 3, padding = 1)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def summary(self):

        model = DenoisingAutoencoder().to(device)
        torchsummary.summary(model, self.input_shape[::-1])
    

class DenoisingCNN(nn.Module):

    def __init__(self, input_shape = (100, 1)):
        super().__init__()
        self.input_shape = input_shape

        ## More like a image-to-image translation
        self.dcnn = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels = 32, out_channels = 1, kernel_size=3, padding=1),
        )

    def forward(self, x):

        output = self.dcnn(x)
        return output
    
    def summary(self):

        model = DenoisingCNN().to(device)
        torchsummary.summary(model, self.input_shape[::-1])
    

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, return_loss=False, patience=5):
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ## gradient_tracking_activated
    model.train()
    train_loss_list, val_loss_list = [], []

    for epoch in range(num_epochs):
        
        epoch_loss = 0.0 # batch-wise loss calculation
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        best_loss = float("inf")
        epochs_without_improvement = 0  # For early stopping
        
        for noisy, clean in progress_bar:
            
            optimizer.zero_grad()
            
            noisy, clean = noisy.reshape(noisy.shape[0], 1, -1), clean.reshape(clean.shape[0], -1, 1)
            
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy).reshape(*clean.shape)
            
            loss = criterion(outputs, clean)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.reshape(noisy.shape[0], 1, -1), clean.reshape(clean.shape[0], -1, 1)
                noisy, clean = noisy.to(device), clean.to(device)
                
                outputs = model(noisy).reshape(*clean.shape)
                loss = criterion(outputs, clean)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    if return_loss:
        return model, train_loss_list, val_loss_list
    else:
        return model

if __name__ == '__main__':
    print('Running ... __nn_based__.py now ...')