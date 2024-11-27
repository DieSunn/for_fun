import os
import sys
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox

# Определяем путь к файлу модели
def get_model_path():
    if getattr(sys, 'frozen', False):  # Если это exe
        return os.path.join(sys._MEIPASS, "even_odd_model.pth")
    else:
        return "even_odd_model.pth"

# Определение архитектуры модели
class EvenOddModel(nn.Module):
    def __init__(self):
        super(EvenOddModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Загрузка модели
model = EvenOddModel()
model.load_state_dict(torch.load(get_model_path()))
model.eval()


# Функция для проверки чётности
def check_even_or_odd(number):
    with torch.no_grad():
        input_number = torch.tensor([[number]], dtype=torch.float32)
        prediction = model(input_number).item()
        return "Чётное" if prediction < 0.5 else "Нечётное"


# Функция для обработки ввода
def on_generate():
    try:
        # Сброс текста метки результата
        result_label.config(text="")

        # Получаем число из текстового поля
        number = int(entry.get())

        # Проверяем чётность
        result = check_even_or_odd(number)

        # Отображаем результат в метке
        result_label.config(text=f"Результат: {result}")
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректное число!")


# Создание окна приложения
root = tk.Tk()
root.title("Чётность числа")
root.geometry("300x200")

# Поле ввода
entry_label = tk.Label(root, text="Введите число:")
entry_label.pack(pady=5)

entry = tk.Entry(root, width=20)
entry.pack(pady=5)

# Кнопка
generate_button = tk.Button(root, text="Сгенерировать", command=on_generate)
generate_button.pack(pady=10)

# Метка для результата
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Запуск приложения
root.mainloop()
