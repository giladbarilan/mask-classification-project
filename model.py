import keras.layers as layers
import keras.models as models
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras.callbacks import ModelCheckpoint
from sys import argv
from data_reciever import DataReceiver

img_width, img_height = 150, 150
BATCH_SIZE = 32
NB_CLASSES = 4  # 3 # has mask, has half mask, has no mask
history = None

'''
# want to save the state of global_train, test and validation
receiver = DataReceiver(getcwd())
train, test, validation = receiver.get_train_test_validation()  # receive images path of train, test, validation.
'''


def train_model():
    global history

    train, validation = DataReceiver.receive_for_level('train', BATCH_SIZE), DataReceiver.receive_for_level('validation', BATCH_SIZE)
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(NB_CLASSES, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    """
    model.fit(np.array(train[0]), np.asarray(train[1]).astype(np.uint8), epochs=30, batch_size=64,
              validation_data=(np.array(validation[0]), np.asarray(validation[1]).astype(np.uint8)))
    """
    filepath = "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    best_filepath = 'best-saved-model-{epoch:02d}-{val_loss:.2f}.hdf5'
    best_model_checkpoint = ModelCheckpoint(best_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit(train[0], steps_per_epoch=train[1], validation_steps=validation[1],
                        epochs=16, validation_data=validation[0], verbose=1, callbacks=[checkpoint, best_model_checkpoint])

    return model
    # score = model.evaluate(np.array(test[0]), np.asarray(test[1]).astype(np.uint8), verbose=1)
    # print(score)


def evaluate_model(model):
    test = DataReceiver.receive_for_level('test', BATCH_SIZE)
    return model.evaluate(test[0], steps=test[1], verbose=1)


def draw_model_graph():
    global history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'accuracy-{time.time()}.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'loss-{time.time()}.png')
    plt.show()


if __name__ == '__main__':
    if len(argv) < 2:
        TEMP_MODEL = train_model()
        TEMP_MODEL.save(f'temp_model-{time.time()}/')
        evaluate_model(TEMP_MODEL)
        draw_model_graph()
