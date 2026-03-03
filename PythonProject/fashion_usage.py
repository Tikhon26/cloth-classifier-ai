# import tensorflow as tf
# from tensorflow.keras.datasets import fashion_mnist
# import matplotlib.pyplot as plt
# import random
#
# model = tf.keras.models.load_model('my_model.keras')
#
# (x_train,_), (x_test, y_test) = fashion_mnist.load_data()
# x_test = x_test / 255.0
# x_test = x_test.reshape(-1, 28 * 28)
#
# class_names = ["T-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
#
# def predict_image(index):
#     img = x_test[index]
#     img_expanded = img.reshape(1, 28, 28, 1)
#     predictions = model.predict(img_expanded)
#     predicted_class = predictions[0].argmax()
#     plt.imshow(img, cmap = 'gray')
#     plt.title(f"Предсказано: {class_names[predicted_class]}\n Истинный класс: {class_names[y_test[index]]}")
#     plt.axis('off')
#     plt.show()
#
# random_index = random.randint(0, len(x_test) - 1)
# predict_image(random_index)


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random

# Загрузка модели
model = tf.keras.models.load_model('my_model.keras')

# Загрузка данных
(x_train, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255.0

class_names = ["T-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]


def predict_image(index):
    img = x_test[index]
    img_expanded = img.reshape(1, 28, 28, 1)  # Изменено для модели
    predictions = model.predict(img_expanded)
    predicted_class = predictions[0].argmax()

    plt.imshow(img, cmap='gray')  # Теперь img - это (28, 28)
    plt.title(f"Предсказано: {class_names[predicted_class]}\n Истинный класс: {class_names[y_test[index]]}")
    plt.axis('off')
    plt.show()


# Вызов функции для случайного индекса
if __name__ == "__main__":
    random_index = random.randint(0, len(x_test) - 1)
    predict_image(random_index)