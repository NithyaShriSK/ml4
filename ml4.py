
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


digits = load_digits()
X = digits.data / 16.0
y = to_categorical(digits.target, 10)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    Dense(64, activation='relu', input_shape=(64,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", round(accuracy * 100, 2), "%")


index = 0
image = X_test[index].reshape(1, 64)

prediction = model.predict(image, verbose=0)
predicted_digit = np.argmax(prediction)
actual_digit = np.argmax(y_test[index])

print("Actual Digit:", actual_digit)
print("Predicted Digit:", predicted_digit)


plt.imshow(image.reshape(8, 8), cmap='gray')
plt.title("Predicted Digit: " + str(predicted_digit))
plt.axis('off')
plt.show()
