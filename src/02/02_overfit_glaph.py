import matplotlib.pyplot as plt

plt.subplot(1,2, 1)
plt.ylim(0, 1)
plt.plot(history1.history['acc'], label="acc", color="red", linestyle='dotted')
plt.plot(history1.history['val_acc'], label="val_acc", color="red")
plt.legend()

plt.subplot(1,2, 2)
plt.ylim(0, 1)
plt.plot(history1.history['loss'], label="loss", color="red", linestyle='dotted')
plt.plot(history1.history['val_loss'], label="val_loss", color="red")
plt.legend()

plt.show()
