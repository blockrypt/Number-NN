import numpy as np
import matplotlib.pyplot as plt
import utils

images, labels = utils.load_dataset()

# Инициализация весов и смещений
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_input_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 3
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Epoch №{epoch}")
    e_loss = 0
    e_correct = 0
    num_samples = images.shape[0]

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Прямой проход
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))

        output_raw = bias_hidden_to_output + weights_input_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        # Кросс-энтропийная потеря
        e_loss += -np.sum(label * np.log(output + 1e-9))

        # Оценка точности
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Обратный проход
        delta_output = output - label
        weights_input_to_output -= learning_rate * delta_output @ hidden.T
        bias_hidden_to_output -= learning_rate * delta_output

        delta_hidden = (weights_input_to_output.T @ delta_output) * (hidden * (1 - hidden))
        weights_input_to_hidden -= learning_rate * delta_hidden @ image.T
        bias_input_to_hidden -= learning_rate * delta_hidden

    print(f"Loss: {round((e_loss / num_samples), 3)}")
    print(f"Accuracy: {round((e_correct / num_samples) * 100, 3)}%")

# Тестирование модели
import random

test_image = random.choice(images)
image = np.reshape(test_image, (-1, 1))
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))

output_raw = bias_hidden_to_output + weights_input_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the number is: {output.argmax()}")
plt.savefig('NN.png')
