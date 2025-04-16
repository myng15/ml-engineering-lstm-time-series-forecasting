import torch
import torch.nn as nn
import numpy as np


def train_model(
        model, 
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        num_epochs=100, 
        learning_rate=1e-4, 
        device="cpu"
    ):
    model = model.to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
 
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        if X_test is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = criterion(y_test_pred, y_test)
            
            test_losses.append(test_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        elif epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

    return model.eval(), train_losses, test_losses


