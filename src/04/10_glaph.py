import matplotlib.pyplot as pltm

plt.ylim(0.5, 1)
plt.plot(history.history['acc'], label="acc")
plt.plot(history.history['val_acc'], label="val_acc")
plt.legend()

plt.show()
