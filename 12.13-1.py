from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
(trainX, trainy), (testX, testy) = mnist.load_data()
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
print('Means train=%.3f, test=%.3f' % (trainX.mean(), testX.mean()))
datagen = ImageDataGenerator(featurewise_center=True)
datagen.fit(trainX)
print('Data Generator Mean: %.3f' % datagen.mean)
iterator = datagen.flow(trainX, trainy, batch_size=64)
batchX, batchy = iterator.next()
print(batchX.shape, batchX.mean())
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
batchX, batchy = iterator.next()
print(batchX.shape, batchX.mean())