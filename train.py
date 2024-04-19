import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, save=True, verbose=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lossTrain, lossVal = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = round(running_loss/len(train_loader.dataset), 5)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = round(val_loss/len(val_loader.dataset), 5)
        if verbose:
            print(f"Epoch {epoch+1}\nTrain Loss: {avg_train_loss}; Val Loss: {avg_val_loss}")
        lossTrain.append(avg_train_loss)
        lossVal.append(avg_val_loss)
    if save:
        formatted_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M")
        torch.save(model.state_dict(), 'ckpts/{}-{}.pth'.format(model.name, formatted_datetime))
        print("saving to", 'ckpts/{}-{}.pth'.format(model.name, formatted_datetime))
    
    return lossTrain, lossVal