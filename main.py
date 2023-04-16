import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from cnn import ImageClassificationModel


train_dir = './train_set/'

test_dir = './test_set/'

# Use ImageDataGenerator for data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

'''
rescale: A scaling factor for the pixel values of the input images. In this case, the pixel values are rescaled to be between 0 and 1 by dividing them by 255.
rotation_range: A range, in degrees, within which to randomly rotate the input images.
width_shift_range and height_shift_range: Ranges within which to randomly shift the width and height dimensions of the input images, as a fraction of the total width and height.\
shear_range: A range, in radians, within which to randomly apply shearing transformations to the input images.
zoom_range: A range, as a fraction of the original size, within which to randomly zoom into the input images.
horizontal_flip: A boolean indicating whether to randomly flip input images horizontally.
fill_mode: The strategy for filling in newly created pixels that may have been introduced during image transformations, such as rotation or shifting. In this case, the strategy is to fill in the nearest pixel value.
'''

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = train_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

if __name__ == '__main__':
    data, lables = next(train_generator)
    print(data.shape)  # (20, 128, 128, 3)
    print(lables.shape)  # (20,)
    img_test = Image.fromarray((255 * data[0]).astype('uint8'))
    img_test.show()
    print(lables[0])

    # build our model from cnn.py
    model = ImageClassificationModel()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=500,
        validation_data=test_generator,
        validation_steps=test_generator.samples/test_generator.batch_size
    )

    # print the acc of our model on test set
    test_eval = model.evaluate_generator(test_generator)
    print(test_eval)

    # save the model
    model.save('./savedmodel/ourmodel')

    # plot and save the figures for 'Training and validation accuracy' and 'Training and validation loss'
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss.png')

