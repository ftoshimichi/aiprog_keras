from keras.utils import plot_model

# 変数modelは学習済みのモデル
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
