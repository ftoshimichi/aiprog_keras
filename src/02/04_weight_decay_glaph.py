import matplotlib.pyplot as plt

plt.ylim(0, 1)
plt.plot(history1.history['acc'], label="acc", color="red", linestyle='dotted')
plt.plot(history1.history['val_acc'], label="val_acc", color="red")
plt.plot(history2.history['acc'], label="wd_acc", color="blue", linestyle='dotted')
plt.plot(history2.history['val_acc'], label="wd_val_loss", color="blue")
plt.legend()

plt.show()
