from keras import backend as K
import keras
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,Cropping2D,Lambda,GlobalAveragePooling2D,BatchNormalization,AveragePooling2D
from keras.models import Sequential,Model
from keras.optimizers import SGD,rmsprop
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.python.ops import array_ops

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def hn_multilabel_loss(y_true, y_pred):
    y_pred = K.clip (y_pred, K.epsilon(), 1 - K.epsilon())
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=0.75):
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)



def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

from sklearn.model_selection import train_test_split

train_df,test_df=train_test_split(train_csv,test_size=TEST_SIZE)


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss

#df_feature=np.array(df_feature)

# df_label=np.array(df_label)

#df_feature = preprocess_input(df_feature) ## Rescaling w.r.t. 'imagenet' dataset.

# x_train, x_test, y_train, y_test = train_test_split(df_feature, df_label, test_size=TEST_SIZE)

callbacks = [
            EarlyStopping(patience=15, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
            ]
## Pretraining with imagenet weights

base_model = DenseNet121(include_top=False, weights='imagenet',
                         input_shape=(SIZE, SIZE, 3), classes=15)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.2)(x)

predictions = Dense(4, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

## Freeze network except last three layers:

for layer in model.layers:
    layer.trainable = False

for i in range(-3,0):
    model.layers[i].trainable = True

model.compile(optimizer=Adam(lr=1e-4, decay=1e-9), loss=hn_multilabel_loss, metrics=['binary_crossentropy',
              categorical_crossentropy, categorical_accuracy, top_3_accuracy])
from keras.preprocessing.image import ImageDataGenerator

# Construct the training image generator for data augmentation

aug = ImageDataGenerator(
         rotation_range = 320,
         width_shift_range = 0.45,
         height_shift_range = 0.25,
         shear_range = 0.40,
         zoom_range = 0.30,
         horizontal_flip = True,
         vertical_flip = True,
         # preprocess_input=True,
         fill_mode ='nearest'
        )

aug_val = ImageDataGenerator(rotation_range = 20,
         width_shift_range = 0.15,
         height_shift_range = 0.25,
         shear_range = 0.20,
         zoom_range = 0.20,
         horizontal_flip = True,
         vertical_flip = True,
         # preprocess_input=True,
         fill_mode ='nearest')

# Train the network
train_generator=aug.flow_from_dataframe(dataframe=train_df,
                                            directory="./",
                                            x_col="ImageName",
                                            y_col="labels",
                                            subset="training",
                                            batch_size=BS,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(SIZE,SIZE))

valid_generator=aug_val.flow_from_dataframe(dataframe=test_df,
                                            directory="./",
                                            x_col="ImageName",
                                            y_col="labels",
                                            subset="training",
                                            batch_size=BS,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(SIZE,SIZE))
H = model.fit_generator(train_generator,
                        validation_data = valid_generator ,
                        validation_steps = (len(test_df))// BS,
                        callbacks = callbacks,
                        steps_per_epoch = len(train_df) // BS,
                        epochs = 5)
## Save pretraining model.

model.save_weights('model_PRE.h5')
del model

## Training & fine-tuning without imagenet weights

base_model = DenseNet121(include_top=False, weights=None,
                         input_shape=(SIZE, SIZE, 3), classes=15)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(4, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('model_PRE.h5')

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(lr=1e-5, decay=1e-9), loss=hn_multilabel_loss, metrics=['binary_crossentropy',
              categorical_crossentropy, categorical_accuracy, top_3_accuracy])
# Construct the training image generator for data augmentation

H = model.fit_generator(train_generator,
                        validation_data = valid_generator ,
                        validation_steps = (len(test_df))// BS,
                        callbacks = callbacks,
                        steps_per_epoch = len(train_df) // BS,
                        epochs = 10)

model.save_weights('model_weights.h5')
import pickle
output_dict={train_generator.class_indices[label]:label for label in train_generator.class_indices.keys()}
with open("output_dict.pkl","wb") as file:
    pickle.dump(output_dict,file)
