import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


def create_dataloaders(X, y, batch_size=16, train_ratio=0.7, val_ratio=0.2):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # 按时间顺序划分（禁止打乱！）
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 转为张量
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, (X_test, y_test)


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=1, output_horizon=15, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention 机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 解码器：将上下文向量映射到15天预测
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_horizon)
        )

    def forward(self, x):
        # x: (batch_size, seq_len=45, input_dim=7)
        lstm_out, _ = self.lstm(x)  # (batch, 45, hidden_dim)

        # 计算注意力权重
        attn_weights = self.attention(lstm_out)  # (batch, 45, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 归一化

        # 加权求和得到上下文向量
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)

        # 预测未来15天
        output = self.decoder(context)  # (batch, 15)
        return output


def mape_loss(y_true, y_pred):
    """计算 MAPE（避免除零）"""
    return torch.mean(torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-6))) * 100


def train_model(model, train_loader, val_loader, device, epochs=200, patience=15):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_mape = float('inf')
    trigger_times = 0
    history = {'train_loss': [], 'val_mape': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.L1Loss()(y_pred, y_batch)  # MAE loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_mape = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                val_mape += mape_loss(y_batch, y_pred).item()
        val_mape /= len(val_loader)

        # 记录
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_mape'].append(val_mape)

        # 学习率调度
        scheduler.step(val_mape)

        # 早停
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            trigger_times = 0
            torch.save(model.state_dict(), 'best_attention_lstm.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val MAPE: {val_mape:.2f}%")

    model.load_state_dict(torch.load('best_attention_lstm.pth'))
    return model, history