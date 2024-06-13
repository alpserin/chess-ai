# train_model.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model.model import parse_pgn, extract_features, ChessDataset, ChessNN
from sklearn.model_selection import train_test_split

data_path = "datasets/lichess_elite_2022_04.pgn"
print(f"Parsing PGN file: {data_path}")
games = parse_pgn(data_path)
data = [extract_features(game) for game in games]

df = pd.DataFrame(data, columns=['WhiteElo', 'BlackElo', 'Result', 'Moves'])
train_df, val_df = train_test_split(df, test_size=0.2)

train_dataset = ChessDataset(train_df)
val_dataset = ChessDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ChessNN().cuda()
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 20
print_interval = 10

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (features, moves, labels) in enumerate(train_loader):
        features, moves, labels = features.cuda(), moves.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(features.float(), moves.float()).squeeze(1)
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / print_interval:.4f}')
            running_loss = 0.0
    
    scheduler.step()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, moves, labels in val_loader:
            features, moves, labels = features.cuda(), moves.cuda(), labels.cuda()
            outputs = model(features.float(), moves.float()).squeeze(1)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}')

print("Training finished.")
torch.save(model.state_dict(), "chess_model.pth")
print("Model saved to chess_model.pth")
