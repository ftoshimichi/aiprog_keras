import matplotlib.pyplot as plt

plt.ylim(0, 1)
plt.plot(history.history['acc'], label="acc")
plt.plot(history.history['val_acc'], label="val_acc")
plt.legend()

plt.show()
