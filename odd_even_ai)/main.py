import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Генерация данных
def generate_data(num_samples=10000):
    x = torch.randint(0, 1000, (num_samples, 1), dtype=torch.float32)  # Случайные числа
    y = (x % 2).long()  # Метка: 0 для чётных, 1 для нечётных
    return x, y

# Подготовка данных
x_data, y_data = generate_data()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Создание модели
class EvenOddModel(nn.Module):
    def __init__(self):
        super(EvenOddModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)  # Входной слой на 16 нейронов
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # Выходной слой
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = EvenOddModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(x_train.size(0))  # Перемешивание данных
    for i in range(0, x_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices].float()

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Приведение меток к размеру предсказаний
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Оценка на тестовых данных
model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_predictions = (test_outputs > 0.5).squeeze().long()
    accuracy = (test_predictions == y_test.squeeze()).float().mean()
    print(f"Точность на тестовых данных: {accuracy * 100:.2f}%")

# Функция проверки чётности
def check_even_or_odd(number):
    model.eval()
    with torch.no_grad():
        input_number = torch.tensor([[number]], dtype=torch.float32)
        prediction = model(input_number).item()
        return "Чётное" if prediction < 0.5 else "Нечётное"

# Пример использования
number = 42
print(f"Число {number} является {check_even_or_odd(number)}.")
# Сохранение обученной модели
torch.save(model.state_dict(), "even_odd_model.pth")
