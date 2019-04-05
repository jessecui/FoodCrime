import pandas as pd
import numpy as np
import cv2
from keras import optimizers
from sklearn.model_selection import train_test_split
import photocnn

photos_df = pd.read_json('yelp_dataset/photo.json', lines=True)
business_df = pd.read_json('yelp_dataset/business.json', lines=True)

criteria = photos_df['label'] == 'outside'
ophotos_df = photos_df[criteria]

busIds = (list(ophotos_df.business_id))
critB = business_df['business_id'].isin(busIds)
critZ = business_df['postal_code'].isin(['89108', '89110', '89129',
                                         '89102', '89149'])

newbus_df = business_df[critB & critZ]
uphotos_df = ophotos_df[ophotos_df['business_id'].isin(list(newbus_df.business_id))]

Y = []
images = []

for n in range(len(uphotos_df.index)):
    busId = uphotos_df.iloc[n]['business_id']
    photoId = uphotos_df.iloc[n]['photo_id']
    path = 'yelp_dataset/photos/' + photoId + ".jpg"
    image = cv2.imread(path)
    image = cv2.resize(image, (64, 64))
    images.append(image)
    zip = newbus_df[newbus_df['business_id'] == busId]['postal_code'].values[0]
    if zip == '89108':
        Y.append(62.6)
    elif zip == "89110":
        Y.append(56.6)
    elif zip == "89129":
        Y.append(41.1)
    elif zip == "89102":
        Y.append(71.9)
    elif zip == "89149":
        Y.append(38.2)

finalImages = np.array(images) / 255.0
finalY = np.array(Y) / 71.9

model = photocnn.create_cnn(64, 64, 3, regress=True)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(finalImages, finalY, validation_split=0.25, epochs=200, batch_size=8)
preds = model.predict(finalImages)
diff = preds.flatten() - finalY
percentDiff = (diff / finalY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print(mean)
