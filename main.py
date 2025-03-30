# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 18:56:46 2025

@author: martin vojtíšek
"""

#importing required modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import BatchNormalization,LeakyReLU,Conv2D, MaxPooling2D, Flatten, Dense, GaussianNoise, GlobalAveragePooling2D,Dropout,Dropout, Input, Activation, Add, Multiply
from tensorflow.keras import regularizers, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.applications import EfficientNetB0, EfficientNetV2S, EfficientNetV2M
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import sys

#my modules
import utils as MVU

my_map = 'Greens' #for plotting - pallete, so that the same color scheme is used throughout 

suf = 'TL_V2S_2DLs_whole' #I want all files for one training run stored in one folder
#at the moment, script will terminate with error if this folder does not exists.
#this is intentional now, but probably will be changed later

#source data
image_folder='./files/train/'
train_path = './files/Training_set.csv'
test_path = './files/Testing_set.csv'

classes_total=75
train_epochs_max=100
leaky = 'relu' #LeakyReLU(negative_slope=0.2) #or 'relu' for relu - this is the activation func in layers
RunType = 'Transfer' #Train/Load/Transfer - training my model, or loading model and printing graphs or training efficientnet or such
LoadModelIncludesScaling=True #if i load stored efficientNet model, i need to use correct generators without scaling
retrain_whole = True #False/True for IsTrainable parameter in calling library models such as EfficientNetV2S. if False, only new classification layers are trained. 
MVU.check_GPU_support()


train_str = (
"""
      _______   _____               _____   _   _ 
     |__   __| |  __ \      /\     |_   _| | \ | |
        | |    | |__) |    /  \      | |   |  \| |
        | |    |  _  /    / /\ \     | |   | . ` |
        | |    | | \ \   / ____ \   _| |_  | |\  |
        |_|    |_|  \_\ /_/    \_\ |_____| |_| \_|

""")                                                  
 
load_str = (
"""
      _         ____               _____  
     | |       / __ \      /\     |  __ \ 
     | |      | |  | |    /  \    | |  | |
     | |      | |  | |   / /\ \   | |  | |
     | |____  | |__| |  / ____ \  | |__| |
     |______|  \____/  /_/    \_\ |_____/ 

""")

transfer_str = (
"""

      _______   _____               _   _    _____   ______   ______   _____  
     |__   __| |  __ \      /\     | \ | |  / ____| |  ____| |  ____| |  __ \ 
        | |    | |__) |    /  \    |  \| | | (___   | |__    | |__    | |__) |
        | |    |  _  /    / /\ \   | . ` |  \___ \  |  __|   |  __|   |  _  / 
        | |    | | \ \   / ____ \  | |\  |  ____) | | |      | |____  | | \ \ 
        |_|    |_|  \_\ /_/    \_\ |_| \_| |_____/  |_|      |______| |_|  \_\

""")

print_run_type = ''
print("Run type is: ")
match RunType:
    case 'Train':
        print_run_type = train_str
    case 'Load':
        print_run_type = load_str
    case 'Transfer':
        print_run_type = transfer_str
        
print(print_run_type)


train_df = pd.read_csv(train_path)
train_df = train_df[train_df['filename'].str.startswith('Image')] #☺at the end, we are only using the original data
#folder and csv include images downloaded from the internat later, but performance was much worse,
#no time now to solve, therefore reverting back to only original dataset

#kaggle_df = pd.read_csv(test_path) #unused at the moment


#exploratory analysis
MVU.exploratory_report(train_df,suf)


#splitting dataset to training and validation sets
X_train, X_val = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], shuffle=True, random_state=42) #dávat random_state aby to bylo reprodukovatelné

#storing datasets for current training run, in case I want to use the same data and compare models
X_train.to_csv("./files/"+suf+"/X_train.csv")
X_val.to_csv("./files/"+suf+"/X_val.csv")


#ensure all classes in validation set
print(X_val[['label']].value_counts())

#image preprocessing
#i could check if all images are 8 bit color depth, using convert_to_eightbit function from utils module, 
#which is in the same folder as the main.py file.

#easiest way to the image augmentation is using hte ImageDataGenerator class from keras
#using all sorts of augmentation - prevent overfitting
#ImageDataGenerator is deprecated but I dont have time to research different solution
#preparing image generators, training sets have augmentation, validation sets do not
train_generator=MVU.create_image_generator(feed_df=X_train, suffix=suf, prep=True, scaling=True, img_folder=image_folder, IsShuffled=True, storename='training_preprocessing') #batch_size=8 by default, batch parameter can change it
valid_generator=MVU.create_image_generator(feed_df=X_val, suffix=suf, prep=False, scaling=True, img_folder=image_folder, IsShuffled=True)
reprt_generator=MVU.create_image_generator(feed_df=X_val, suffix=suf, prep=False, scaling=True, img_folder=image_folder, IsShuffled=False)
#unfortunately EfficientNetV2 models expect either unscaled images, or -1,1 tensors. so I have to prepare specific generators for transfer learning...
train_generator_t=MVU.create_image_generator(feed_df=X_train, suffix=suf, prep=True, scaling=False, img_folder=image_folder, IsShuffled=True) #batch_size=8 by default, batch parameter can change it
valid_generator_t=MVU.create_image_generator(feed_df=X_val, suffix=suf, prep=False, scaling=False, img_folder=image_folder, IsShuffled=True)
reprt_generator_t=MVU.create_image_generator(feed_df=X_val, suffix=suf, prep=False, scaling=False, img_folder=image_folder, IsShuffled=False)


#callback for training - storing log, early stopping etc
callback_store = "./files/"+suf+"/LastCallback.keras"
checkpoint = ModelCheckpoint(callback_store, monitor='val_accuracy', save_best_only=True)
earlystop = EarlyStopping(monitor='val_accuracy', restore_best_weights=True, min_delta=0.001, patience=10, mode='auto')
csv_logger = CSVLogger("./files/"+suf+"/training_log.csv", append=True)

callback_list = [checkpoint, earlystop, csv_logger]

#steps per epoch - i am setting it to number of batches that would cover the whole dataset once
#it is possible to set it lower or higher
#if i want higher - one epoch would use more samples than dataset, it is a bit more complicated
#either i create bigger dataframe where I concatenate each sample several times
#or create custom generator
spe_t = (train_generator.samples // train_generator.batch_size + 1)
spe_v = (valid_generator.samples // valid_generator.batch_size + 1)
#len(train_generator) je to samé jako spe_t, stejná hodnota

match RunType:
    case 'Train':
        #select our model - various iterations should be stored as diff functions so that code is preserved
        #othervise i would have to use different branches in git
        #then i just select the model I want to train
        model_CNN = MVU.create_compile_re_se_model_02()
        t_gen=train_generator
        v_gen = valid_generator
        r_gen = reprt_generator
    case 'Transfer':
        #select available model from library, rescaling is included in data generator, so excluded from model
        model_CNN = EfficientNetV2S(weights="imagenet", include_top=False)
        t_gen=train_generator_t
        v_gen = valid_generator_t
        r_gen = reprt_generator_t
    case 'Load':
        #in case i want to load some already trained model
        model_CNN = load_model('./files/'+suf+'/my_model.keras')
        if LoadModelIncludesScaling:
            t_gen = train_generator_t
            v_gen = valid_generator_t
            r_gen = reprt_generator_t
        else:
            t_gen = train_generator
            v_gen = valid_generator
            r_gen = reprt_generator
    case _:
        print("Exiting script - invalid RunType parameter")
        sys.exit() #this raises an exception but should be ok for my script

#print model architectury summary on console
model_CNN.summary()


#training model if TrainingType Train or Transfer
match RunType:
    case 'Train':
        trained_model = MVU.training_model(model=model_CNN, train_gen=t_gen, val_gen=v_gen, calls=callback_list, steps_t=spe_t, steps_v=spe_v, eps=train_epochs_max)
        history = trained_model.history
        MVU.plot_training_history(model=history, suffix=suf)
        trained_model.save("./files/"+suf+"/my_model.keras")  # Save model
    case 'Transfer':
        trained_model = MVU.transfer_learn_model(model=model_CNN, train_gen=t_gen, val_gen=v_gen, calls=callback_list, classes=classes_total, eps=train_epochs_max, IsTrainable=retrain_whole)
        history = trained_model.history
        MVU.plot_training_history(model=history, suffix=suf)
        trained_model.save("./files/"+suf+"/my_model.keras")  # Save model
    case 'Load':
        trained_model = model_CNN
        #no history available 
        #no print
        #no save



#performance reports and graphs
testing_all, mislabel_df = MVU.print_classification_report(trained_model, r_gen, suffix=suf)

df_filenames = X_val.reset_index(drop=True)
mislabel_df = mislabel_df.merge(df_filenames, left_index=True, right_index=True)
mislabel_df.to_csv("./files/"+suf+"/misclassified_samples.csv") #save misclassified samples and correcponding filenames


#plot essential graphs
MVU.plot_and_save_essentials(all_df=testing_all, mis_df=mislabel_df, get_cmap=my_map, suffix=suf)




#predictions = model_CNN.predict(reprt_generator)

#entropies = []
#for i in range(0, len(X_val)-1):
#    entropies.append(MVU.calculate_entropy(predictions[i]))

#print(train_df[train_df["label"]=='WOOD SATYR'])

