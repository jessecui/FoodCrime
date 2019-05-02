import pickle
import pandas as pd
import statistics
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import photocnn

pickle_in = open("train_valid_test_business_ids.pk", 'rb')
train_testBus = pickle.load(pickle_in)
pickle_in.close()

pickle_in2 = open("main_data_tuple_2.pk", 'rb')
main_data = pickle.load(pickle_in2)
pickle_in2.close()

pickle_in3 = open("predictions_text_rf.pk", 'rb')
predRFPrior = pickle.load(pickle_in3)
pickle_in3.close()

pickle_in4 = open("xgb_predictions_main.pk", 'rb')
predXGBPrior = pickle.load(pickle_in4)
pickle_in4.close()

inter = main_data[1]
inter2 = main_data[3]
inter3 = main_data[5]

allY_trainViolent = inter[:, 1]
allY_validViolent = inter2[:, 1]
allY_testViolent = inter3[:, 1]

allY_trainProperty = inter[:, 0]
allY_validProperty = inter2[:, 0]
allY_testProperty = inter3[:, 0]

train_business_ids = train_testBus[0]
valid_business_ids = train_testBus[1]
test_business_ids = train_testBus[2]

photos_df = pd.read_json('yelp_dataset/photo.json', lines=True)
criteria = photos_df['label'] == 'outside'
criteriaI = photos_df['label'] == 'inside'
criteriaF = photos_df['label'] == 'food'
criteriaD = photos_df['label'] == 'drink'
ophotos_df = photos_df[criteria]
iphotos_df = photos_df[criteriaI]
fphotos_df = photos_df[criteriaF]
dphotos_df = photos_df[criteriaD]

Y = []
Yp1 = []
Ytest = []
Ytestp1 = []
images = []
imagesTest = []

# outside
for i in range(len(train_business_ids)):
    busId = train_business_ids[i]
    criteria2 = ophotos_df['business_id'] == busId
    targPhotos_df = ophotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        images.append(image)
        Y.append(allY_trainViolent[i])
        Yp1.append(allY_trainProperty[i])

for i in range(len(test_business_ids)):
    busId = test_business_ids[i]
    criteria2 = photos_df['business_id'] == busId
    targPhotos_df = photos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        imagesTest.append(image)
        Ytest.append(allY_testViolent[i])
        Ytestp1.append(allY_testProperty[i])

trainImages = np.array(images) / 255
testImages = np.array(imagesTest) / 255

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(trainImages)

modelO = photocnn.create_cnn(32, 32, 3)
modelO.compile(loss='mean_squared_error', optimizer="adam")
modelO.fit_generator(train_datagen.flow(trainImages, Y, batch_size=32),
                     steps_per_epoch=len(trainImages) / 32, epochs=5, validation_data=(testImages, Ytest))

modelOP = photocnn.create_cnn(32, 32, 3)
modelOP.compile(loss='mean_squared_error', optimizer="adam")
modelOP.fit_generator(train_datagen.flow(trainImages, Yp1, batch_size=32),
                      steps_per_epoch=len(trainImages) / 32, epochs=5, validation_data=(testImages, Ytestp1))

Y2 = []
Yp2 = []
Ytest2 = []
Yptest2 = []
images2 = []
imagesTest2 = []

# inside
for i in range(len(train_business_ids)):
    busId = train_business_ids[i]
    criteria2 = iphotos_df['business_id'] == busId
    targPhotos_df = iphotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        images2.append(image)
        Y2.append(allY_trainViolent[i])
        Yp2.append(allY_trainProperty[i])
for i in range(len(test_business_ids)):
    busId = test_business_ids[i]
    criteria2 = iphotos_df['business_id'] == busId
    targPhotos_df = iphotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        imagesTest2.append(image)
        Ytest2.append(allY_testViolent[i])
        Yptest2.append(allY_testProperty[i])

trainImages2 = np.array(images2) / 255
testImages2 = np.array(imagesTest2) / 255

train_datagen2 = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen2.fit(trainImages2)

modelI = photocnn.create_cnn(32, 32, 3)
modelI.compile(loss='mean_squared_error', optimizer="adam")
modelI.fit_generator(train_datagen2.flow(trainImages2, Y2, batch_size=32),
                     steps_per_epoch=len(trainImages2) / 32, epochs=5, validation_data=(testImages2, Ytest2))

modelIP = photocnn.create_cnn(32, 32, 3)
modelIP.compile(loss='mean_squared_error', optimizer="adam")
modelIP.fit_generator(train_datagen2.flow(trainImages2, Yp2, batch_size=32),
                      steps_per_epoch=len(trainImages2) / 32, epochs=5, validation_data=(testImages2, Yptest2))

Y3 = []
Yp3 = []
Ytest3 = []
Yptest3 = []
images3 = []
imagesTest3 = []
print(len(train_business_ids))
# food
for i in range(len(train_business_ids)):
    print(i)
    busId = train_business_ids[i]
    criteria2 = fphotos_df['business_id'] == busId
    targPhotos_df = fphotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        images3.append(image)
        Y3.append(allY_trainViolent[i])
        Yp3.append(allY_trainProperty[i])
for i in range(len(test_business_ids)):
    busId = test_business_ids[i]
    criteria2 = fphotos_df['business_id'] == busId
    targPhotos_df = fphotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        imagesTest3.append(image)
        Ytest3.append(allY_testViolent[i])
        Yptest3.append(allY_testProperty[i])

trainImages3 = np.array(images3) / 255
testImages3 = np.array(imagesTest3) / 255

train_datagen3 = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen3.fit(trainImages3)
print(len(trainImages3))
print(len(Yp3))

modelF = photocnn.create_cnn(32, 32, 3)
modelF.compile(loss='mean_squared_error', optimizer="adam")
modelF.fit_generator(train_datagen.flow(trainImages3, Y3, batch_size=32),
                     steps_per_epoch=len(trainImages3) / 32, epochs=5, validation_data=(testImages3, Ytest3))

modelFP = photocnn.create_cnn(32, 32, 3)
modelFP.compile(loss='mean_squared_error', optimizer="adam")
modelFP.fit_generator(train_datagen.flow(trainImages3, Yp3, batch_size=32),
                      steps_per_epoch=len(trainImages3) / 32, epochs=5, validation_data=(testImages3, Yptest3))

len(Yp3)

Y4 = []
Yp4 = []
Ytest4 = []
Yptest4 = []
images4 = []
imagesTest4 = []

# drinks
for i in range(len(train_business_ids)):
    busId = train_business_ids[i]
    criteria2 = dphotos_df['business_id'] == busId
    targPhotos_df = dphotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        images4.append(image)
        Y4.append(allY_trainViolent[i])
        Yp4.append(allY_trainProperty[i])
for i in range(len(test_business_ids)):
    busId = test_business_ids[i]
    criteria2 = dphotos_df['business_id'] == busId
    targPhotos_df = dphotos_df[criteria2]
    for h in range(len(targPhotos_df.index)):
        photoId = targPhotos_df.iloc[h]['photo_id']
        path = 'yelp_dataset/photos/' + photoId + ".jpg"
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        imagesTest4.append(image)
        Ytest4.append(allY_testViolent[i])
        Yptest4.append(allY_testProperty[i])

trainImages4 = np.array(images4) / 255
testImages4 = np.array(imagesTest4) / 255

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(trainImages4)

modelD = photocnn.create_cnn(32, 32, 3)
modelD.compile(loss='mean_squared_error', optimizer="adam")
modelD.fit_generator(train_datagen.flow(trainImages4, Y4, batch_size=32),
                     steps_per_epoch=len(trainImages4) / 32, epochs=5, validation_data=(testImages4, Ytest4))

modelDP = photocnn.create_cnn(32, 32, 3)
modelDP.compile(loss='mean_squared_error', optimizer="adam")
modelDP.fit_generator(train_datagen.flow(trainImages4, Yp4, batch_size=32),
                      steps_per_epoch=len(trainImages4) / 32, epochs=5, validation_data=(testImages4, Yptest4))

drinkValScores = []
dValP = []

foodValScores = []
fValP = []

outValScores = []
oValP = []

inValScores = []
iValP = []
print(len(valid_business_ids))
for i in range(len(valid_business_ids)):
    print(i)
    busId = valid_business_ids[i]
    realS = allY_validViolent[i]

    criteria2 = dphotos_df['business_id'] == busId
    drinkPhotos_df = dphotos_df[criteria2]

    criteria2 = fphotos_df['business_id'] == busId
    foodPhotos_df = fphotos_df[criteria2]

    criteria2 = ophotos_df['business_id'] == busId
    outPhotos_df = ophotos_df[criteria2]

    criteria2 = iphotos_df['business_id'] == busId
    inPhotos_df = iphotos_df[criteria2]

    iVD = []
    yD = []

    iVF = []
    yF = []

    iVO = []
    yO = []

    iVI = []
    yI = []
    if (not (drinkPhotos_df.empty)):
        for h in range(len(drinkPhotos_df.index)):
            photoId = drinkPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVD.append(image)
            yD.append(realS)

        finaliVD = np.array(iVD) / 255
        finalyD = np.array(yD)
        predsD = modelD.predict(finaliVD)
        predsDP = modelDP.predict(finaliVD)

        drinkValScores.append(np.mean(predsD))
        dValP.append(np.mean(predsDP))

    if (not (foodPhotos_df.empty)):
        for h in range(len(foodPhotos_df.index)):
            photoId = foodPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVF.append(image)
            yF.append(realS)

        finaliVF = np.array(iVF) / 255
        finalyF = np.array(yF)
        predsF = modelF.predict(finaliVF)
        predsFP = modelFP.predict(finaliVF)

        foodValScores.append(np.mean(predsF))
        fValP.append(np.mean(predsFP))

    if (not (outPhotos_df.empty)):
        for h in range(len(outPhotos_df.index)):
            photoId = outPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVO.append(image)
            yO.append(realS)

        finaliVO = np.array(iVO) / 255
        finalyO = np.array(yO)
        predsO = modelO.predict(finaliVO)
        predsOP = modelOP.predict(finaliVO)

        outValScores.append(np.mean(predsO))
        oValP.append(np.mean(predsOP))

    if (not (inPhotos_df.empty)):
        for h in range(len(inPhotos_df.index)):
            photoId = inPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVI.append(image)
            yI.append(realS)

        finaliVI = np.array(iVI) / 255
        finalyI = np.array(yI)
        predsI = modelI.predict(finaliVI)
        predsIP = modelIP.predict(finaliVI)

        inValScores.append(np.mean(predsI))
        iValP.append(np.mean(predsIP))

rfPredictionsProperty = predRFPrior[0][:, 0]
rfPredictionsViolent = predRFPrior[0][:, 1]

xgbPredictionsProperty = predXGBPrior[0][:, 0]
xgbPredictionsViolent = predXGBPrior[0][:, 1]

fPre = []
fPreP = []

dPre = []
dPreP = []

oPre = []
oPreP = []

iPre = []
iPreP = []

itIndex1 = 0
itIndex2 = 0
itIndex3 = 0
itIndex4 = 0

for i in range(len(valid_business_ids)):
    print(i)
    busId = valid_business_ids[i]
    realS = allY_validViolent[i]

    criteria2 = dphotos_df['business_id'] == busId
    drinkPhotos_df = dphotos_df[criteria2]

    if (drinkPhotos_df.empty):
        dPre.append(np.mean(drinkValScores))
        dPreP.append(np.mean(dValP))
    else:
        dPre.append(drinkValScores[itIndex1])
        dPreP.append(dValP[itIndex1])
        itIndex1 = itIndex1 + 1

    criteria2 = fphotos_df['business_id'] == busId
    foodPhotos_df = fphotos_df[criteria2]

    if (foodPhotos_df.empty):
        fPre.append(np.mean(foodValScores))
        fPreP.append(np.mean(fValP))
    else:
        fPre.append(foodValScores[itIndex2])
        fPreP.append(fValP[itIndex2])
        itIndex2 = itIndex2 + 1

    criteria2 = ophotos_df['business_id'] == busId
    outPhotos_df = ophotos_df[criteria2]

    if (outPhotos_df.empty):
        oPre.append(np.mean(outValScores))
        oPreP.append(np.mean(oValP))
    else:
        oPre.append(outValScores[itIndex3])
        oPreP.append(oValP[itIndex3])
        itIndex3 = itIndex3 + 1

    criteria2 = iphotos_df['business_id'] == busId
    inPhotos_df = iphotos_df[criteria2]

    if (inPhotos_df.empty):
        iPre.append(np.mean(inValScores))
        iPreP.append(np.mean(iValP))
    else:
        iPre.append(inValScores[itIndex4])
        iPreP.append(iValP[itIndex4])
        itIndex4 = itIndex4 + 1

print(len(dPre))

X_V = (np.vstack((rfPredictionsViolent, xgbPredictionsViolent, np.array(fPre), np.array(dPre), np.array(oPre),
               np.array(iPre)))).transpose()

np.shape(X_V)
X_P = (np.vstack((rfPredictionsProperty, xgbPredictionsProperty, np.array(fPreP), np.array(dPreP), np.array(oPreP),
               np.array(iPreP)))).transpose()

# ensemble
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regV = LinearRegression().fit(X_V, allY_validViolent)
regP = LinearRegression().fit(X_P, allY_validProperty)

print(regV.coef_)
v_predsT = regV.predict(X_V)
meanSqErV = mean_squared_error(allY_validViolent, v_predsT)
print(meanSqErV)
p_predsT = regP.predict(X_P)
meanSqErP = mean_squared_error(allY_validProperty, p_predsT)
print(meanSqErP)
drinkValScoresT = []
dValPT = []

foodValScoresT = []
fValPT = []

outValScoresT = []
oValPT = []

inValScoresT = []
iValPT = []

for i in range(len(test_business_ids)):
    busId = test_business_ids[i]

    criteria2 = dphotos_df['business_id'] == busId
    drinkPhotos_df = dphotos_df[criteria2]

    criteria2 = fphotos_df['business_id'] == busId
    foodPhotos_df = fphotos_df[criteria2]

    criteria2 = ophotos_df['business_id'] == busId
    outPhotos_df = ophotos_df[criteria2]

    criteria2 = iphotos_df['business_id'] == busId
    inPhotos_df = iphotos_df[criteria2]

    iVD = []
    yD = []

    iVF = []
    yF = []

    iVO = []
    yO = []

    iVI = []
    yI = []

    if (not(drinkPhotos_df.empty)):
        for h in range(len(drinkPhotos_df.index)):
            photoId = drinkPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVD.append(image)
            yD.append(realS)

        finaliVD = np.array(iVD) / 255
        finalyD = np.array(yD)
        predsD = modelD.predict(finaliVD)
        predsDP = modelDP.predict(finaliVD)

        drinkValScoresT.append(np.mean(predsD))
        dValPT.append(np.mean(predsDP))

    if (not(foodPhotos_df.empty)):
        for h in range(len(foodPhotos_df.index)):
            photoId = foodPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVF.append(image)
            yF.append(realS)

        finaliVF = np.array(iVF) / 255
        finalyF = np.array(yF)
        predsF = modelF.predict(finaliVF)
        predsFP = modelFP.predict(finaliVF)

        foodValScoresT.append(np.mean(predsF))
        fValPT.append(np.mean(predsFP))

    if (not(outPhotos_df.empty)):
        for h in range(len(outPhotos_df.index)):
            photoId = outPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVO.append(image)
            yO.append(realS)

        finaliVO = np.array(iVO) / 255
        finalyO = np.array(yO)
        predsO = modelO.predict(finaliVO)
        predsOP = modelOP.predict(finaliVO)

        outValScoresT.append(np.mean(predsO))
        oValPT.append(np.mean(predsOP))

    if (not(inPhotos_df.empty)):
        for h in range(len(inPhotos_df.index)):
            photoId = inPhotos_df.iloc[h]['photo_id']
            path = 'yelp_dataset/photos/' + photoId + ".jpg"
            image = cv2.imread(path)
            image = cv2.resize(image, (32, 32))
            iVI.append(image)
            yI.append(realS)

        finaliVI = np.array(iVI) / 255
        finalyI = np.array(yI)
        predsI = modelI.predict(finaliVI)
        predsIP = modelIP.predict(finaliVI)

        inValScoresT.append(np.mean(predsI))
        iValPT.append(np.mean(predsIP))

rfPredictionsPropertyT = predRFPrior[1][:, 0]
rfPredictionsViolentT = predRFPrior[1][:, 1]

xgbPredictionsPropertyT = predXGBPrior[1][:, 0]
xgbPredictionsViolentT = predXGBPrior[1][:, 1]

fPreT = []
fPrePT = []

dPreT = []
dPrePT = []

oPreT = []
oPrePT = []

iPreT = []
iPrePT = []

itIndex1 = 0
itIndex2 = 0
itIndex3 = 0
itIndex4 = 0

for i in range(len(test_business_ids)):
    busId = test_business_ids[i]

    criteria2 = dphotos_df['business_id'] == busId
    drinkPhotos_df = dphotos_df[criteria2]

    if (drinkPhotos_df.empty):
        dPreT.append(statistics.mean(drinkValScoresT))
        dPrePT.append(statistics.mean(dValPT))
    else:
        dPreT.append(drinkValScoresT[itIndex1])
        dPrePT.append(dValPT[itIndex1])
        itIndex1 = itIndex1 + 1

    criteria2 = fphotos_df['business_id'] == busId
    foodPhotos_df = fphotos_df[criteria2]

    if (foodPhotos_df.empty):
        fPreT.append(statistics.mean(foodValScoresT))
        fPrePT.append(statistics.mean(fValPT))
    else:
        fPreT.append(foodValScoresT[itIndex2])
        fPrePT.append(fValPT[itIndex2])
        itIndex2 = itIndex2 + 1

    criteria2 = ophotos_df['business_id'] == busId
    outPhotos_df = ophotos_df[criteria2]

    if (outPhotos_df.empty):
        oPreT.append(statistics.mean(outValScoresT))
        oPrePT.append(statistics.mean(oValPT))
    else:
        oPreT.append(outValScoresT[itIndex3])
        oPrePT.append(oValPT[itIndex3])
        itIndex3 = itIndex3 + 1

    criteria2 = iphotos_df['business_id'] == busId
    inPhotos_df = iphotos_df[criteria2]

    if (inPhotos_df.empty):
        iPreT.append(statistics.mean(inValScoresT))
        iPrePT.append(statistics.mean(iValPT))
    else:
        iPreT.append(inValScoresT[itIndex4])
        iPrePT.append(iValPT[itIndex4])
        itIndex4 = itIndex4 + 1

from sklearn.metrics import mean_squared_error

X_VT = (np.vstack(rfPredictionsViolentT, xgbPredictionsViolentT, np.array(fPreT), np.array(dPreT), np.array(oPreT),
                np.array(iPreT))).transpose()

X_PT = (np.vstack(rfPredictionsPropertyT, xgbPredictionsPropertyT, np.array(fPrePT), np.array(dPrePT), np.array(oPrePT),
                np.array(iPrePT))).transpose()

v_preds = regV.predict(X_VT)
meanSqErV = mean_squared_error(allY_testViolent, v_preds)

p_preds = regP.predict(X_PT)
meanSqErP = mean_squared_error(allY_testProperty, p_preds)

print(meanSqErVT)
print(meanSqErPT)
