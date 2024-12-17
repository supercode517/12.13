from keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
(trainX, trainY), (testX, testY) = mnist.load_data()
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
print('train min=%.3f, max=%.3f' %(trainX.min(), trainX.max()))
print('test min=%.3f, max=%.3f' %(testX.min(), testX.max()))
datagen =ImageDataGenerator(rescale=1.0/255.0)
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
batchX, batchy = next(train_iterator)
print('Batch shape=%s, min=%.3f, max=%.3f' %(batchX.shape, batchX.min(), batchX.max()))


