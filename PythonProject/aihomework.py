import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('mnist_model.h5')

def prepare_image(image_path):
  img = Image.open(image_path).convert('L')
  img = img.resize((28, 28), Image.LANCZOS)
  img_array = np.array(img, dtype=np.float32)
  img_array = img_array / 255.0
  img_array = img_array.reshape(1, 784)
  return img_array

def predict_digit(image_path):
  prepared_image = prepare_image(image_path)
  prediction = model.predict(prepared_image)
  predicted_label = np.argmax(prediction)
  img = Image.open(image_path).convert('L')
  plt.imshow(img, cmap='gray')
  plt.title(f"Предсказанная цифра: {predicted_label}")
  plt.axis('off')
  plt.show()
  return predicted_label

image_path = '2.png'
predicted_digit = predict_digit(image_path)
print(f"Предсказанная цифра: {predicted_digit}")