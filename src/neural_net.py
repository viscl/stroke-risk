import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class StrokeMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class NeuralNetClassifier:
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.3,
        lr: float = 0.001,
        patience: int = 10,
        max_epochs: int = 200,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.dropout = dropout
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_tensor = torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.float32)).reshape(-1, 1)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, stratify=y_tensor.numpy(),
            random_state=self.random_state,
        )

        self.model_ = StrokeMLP(self.input_dim, self.dropout)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in range(self.max_epochs):
            self.model_.train()
            optimizer.zero_grad()
            out = self.model_(X_tr)
            loss = criterion(out, y_tr)
            loss.backward()
            optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_out = self.model_(X_val)
                val_loss = criterion(val_out, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_tensor = torch.tensor(X.astype(np.float32))
        with torch.no_grad():
            probs = self.model_(X_tensor).numpy().ravel()
        return np.column_stack([1.0 - probs, probs])

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model_.state_dict(),
                "input_dim": self.input_dim,
                "dropout": self.dropout,
            },
            path,
        )

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        instance = NeuralNetClassifier(
            input_dim=checkpoint["input_dim"],
            dropout=checkpoint["dropout"],
        )
        instance.model_ = StrokeMLP(checkpoint["input_dim"], checkpoint["dropout"])
        instance.model_.load_state_dict(checkpoint["state_dict"])
        instance.model_.eval()
        return instance
