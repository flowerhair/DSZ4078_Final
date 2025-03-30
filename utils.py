# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:18:39 2025

@author: martin vojtíšekd
"""
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import GaussianNoise,Dropout, Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Multiply
from PIL import Image
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from scipy.stats import entropy
import torch
import json

def create_compile_re_se_model_01(leaky='relu', classes=75):
    def bottleneck_residual_block(x, filters, reduction=4):
        reduced_filters = filters // reduction  # Reduce computation
        res = Conv2D(reduced_filters, (1, 1), padding='same', activation=leaky)(x)  # 1x1 Conv
        res = BatchNormalization()(res)
        
        res = Conv2D(reduced_filters, (3, 3), padding='same', activation=leaky)(res)  # 3x3 Conv
        res = BatchNormalization()(res)
        
        res = Conv2D(filters, (1, 1), padding='same')(res)  # Restore original shape
        res = BatchNormalization()(res)
        
        res = Add()([x, res])  # Skip connection
        return Activation(leaky)(res)  

    # Squeeze-and-Excitation Block (Lighter Version)
    def se_block(input_tensor, reduction=16):
        filters = input_tensor.shape[-1]  
        se = GlobalAveragePooling2D()(input_tensor)  
        se = Dense(filters // reduction, activation='relu')(se)  
        se = Dense(filters, activation='sigmoid')(se)  
        return Multiply()([input_tensor, se])  

    # Input layer
    input_layer = Input(shape=(224, 224, 3))

    # Initial Conv Layer
    x = Conv2D(64, (3, 3), padding='same', activation=leaky)(input_layer)
    x = BatchNormalization()(x)

    # Bottleneck Residual + SE Block 1
    x = bottleneck_residual_block(x, 64)
    x = se_block(x)

    # Bottleneck Residual + SE Block 2
    x = Conv2D(128, (3, 3), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 128)
    x = se_block(x)

    # Bottleneck Residual + SE Block 3
    x = Conv2D(256, (3, 3), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 256)
    x = se_block(x)
    x = Dropout(0.1)(x)

    # Bottleneck Residual + SE Block 4
    x = Conv2D(512, (3, 3), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 512)
    x = se_block(x)

    # Global Average Pooling - instead of Flatten
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = Dense(512, activation=leaky)(x)  # Reduced size
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation=leaky)(x)  # Reduced size
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    # Output Layer
    output_layer = Dense(classes, activation='softmax')(x)

    # Create model
    model_CNN = Model(inputs=input_layer, outputs=output_layer)

    # Compile Model - lower learning rate
    model_CNN.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    return model_CNN


def create_compile_model_01(act_conv='relu', act_dense='relu', classes=75):
    """
    building the basic convolution sequential model    

    Parameters
    ----------
    act_conv : string or activation, optional
        DESCRIPTION. The default is 'relu'.
    act_dense : string or activation, optional
        DESCRIPTION. The default is 'relu'.
    classes : int, optional
        number of classes for final softmax layer. The default is 75.

    Returns
    -------
    model_CNN : Model
        DESCRIPTION.

    """
    model_CNN = Sequential([
        Input(shape=(224, 224, 3)),
        GaussianNoise(stddev=0.05),
        Conv2D(64, (5, 5), padding='same', activation=act_conv), #was 32
        BatchNormalization(),
        Conv2D(128, (5, 5), padding='same', strides=2, activation=act_conv), #was 64
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation=act_conv),
        BatchNormalization(),
        Dropout(0.15),
        Conv2D(256, (3, 3), padding='same', strides=1, activation=act_conv),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=2, activation=act_conv),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', strides=2, activation=act_conv),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(512, activation=act_dense),
        BatchNormalization(),
        Dropout(0.1),
        Dense(384, activation=act_dense),
        BatchNormalization(),
        Dropout(0.1),
        Dense(256, activation=act_dense),
        BatchNormalization(),
        Dense(classes, activation='softmax')
    ])

    #compile model
    #didnt really set anything here, mostly defaults
    #loss is categorical_crossentropy because it is a classification problem
    #for metric, accuracy is selected, it is easier to interpret

    model_CNN.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model_CNN

def exploratory_report(train_df,suffix):
    
    #celkem dat
    print(train_df['label'].count())

    #check for missing labels
    print(f"NANs: {train_df['label'].isna().sum()}")

    #how many different labels?
    print(f"Categories: {len(train_df['label'].unique())}")
    
    avg_labels = train_df['label'].count() / len(train_df['label'].unique())

    #plotting label distribution - is it balanced?
    plt.figure(figsize=(16, 8))
    sns.countplot(data=train_df, x='label', color='lightsalmon', linewidth=1, edgecolor='black', gap=0.2)
    plt.axhline(y=avg_labels, color='maroon', linewidth=1.5, linestyle='dashed')
    plt.xticks(rotation=90)

    plt.title('Distribution of Butterfly Classes')
    plt.xlabel('Butterfly Classes')
    plt.ylabel('Number of Images')

    plt.tight_layout()
    plt.savefig("./files/"+suffix+"/ClassDistro.png", format="png", dpi=300, bbox_inches="tight")  # High-res PNG
    #plt.show()
    
    

def check_torch_support():
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.device_count())  # Should show number of GPUs
    print(torch.cuda.get_device_name(0)) 

def create_image_generator(feed_df, suffix, prep=True, scaling=True, img_folder='./files/train/', IsShuffled=True, batch=8, storename='preprocessing'):
    
    if (scaling is True):
        rescale_factor = 1.0/255
    else:
        rescale_factor = 1.0
            
    if(prep is True):
        img_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.8,1.20),
        shear_range=0.15,
        zoom_range=0.15,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=rescale_factor,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
        )
    else:
        #for validation data i am not using any augmentation
        img_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None   ,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=rescale_factor,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
        )

    image_generator = img_gen.flow_from_dataframe(
    directory=img_folder,
    dataframe=feed_df,
    x_col='filename',
    y_col='label',
    target_size=(224,224), #♦resizing all images to the same size
    batch_size=batch, #how many images are loaded at a time for processing
    shuffle=IsShuffled,
    class_mode='categorical'    
    )
        
    prep_dict={    
        'featurewise_center':False,
        'samplewise_center':False,
        'featurewise_std_normalization':False,
        'samplewise_std_normalization':False,
        'zca_whitening':False,
        'zca_epsilon':1e-06,
        'rotation_range':40,
        'width_shift_range':0.2,
        'height_shift_range':0.2,
        'brightness_range':(0.8,1.20),
        'shear_range':0.15,
        'zoom_range':0.15,
        'channel_shift_range':0.0,
        'fill_mode':'nearest',
        'cval':0.0,
        'horizontal_flip':True,
        'vertical_flip':True,
        'rescale':rescale_factor,
        'preprocessing_function':None,
        'data_format':None,
        'validation_split':0.0,
        'interpolation_order':1
        }
    with open("./files/"+suffix+"/"+storename+".json", "w") as f:
        json.dump(prep_dict, f)
    return image_generator







def calculate_entropy(prediction):
    return entropy(prediction, base=2)

def print_classification_report(model, valid_gen, suffix):
    #prints classification report, confusion matrix
    #and also returns dataframe of all results and also mislabelled samples
    
    # Assuming you have a trained CNN model
    saved_model = model  # Load your trained model
    
    # Assuming you have an image generator
    image_generator = valid_gen  # Your image generator (e.g., validation/test generator)
    class_names = list(image_generator.class_indices.keys())  # Get class names
    
    # Get true labels and predicted labels
    y_true = image_generator.classes  # True labels
    y_pred_proba = saved_model.predict(image_generator)  # Predicted probabilities
    y_pred = np.argmax(y_pred_proba, axis=1)  # Convert to class indices
    
    # Create DataFrame to analyze predictions
    df_results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_label": [class_names[i] for i in y_true],
        "y_pred_label": [class_names[i] for i in y_pred]
    })
    
    # Display misclassified samples
    df_misclassified = df_results[df_results["y_true"] != df_results["y_pred"]]
    print("Misclassified Samples:\n", df_misclassified)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(30, 30))  # Make it HUGE for visibility
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # Save as vector format (SVG, PDF) and high-resolution PNG
    plt.savefig("./files/"+suffix+"/confusion_matrix.svg", format="svg")  # Vector format
    #plt.savefig("confusion_matrix.pdf", format="pdf")  # Vector format
    plt.savefig("./files/"+suffix+"/confusion_matrix.png", format="png", dpi=300, bbox_inches="tight")  # High-res PNG
    
    #plt.show()
    output_file = open("./files/"+suffix+"/ClassificationReport.txt", 'w')
    # Print classification report
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names), file=output_file)
    output_file.close()
    return df_results, df_misclassified


def check_GPU_support():
    #prints output of gpu availability tests    
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Is built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))

def convert_to_eightbit(in_folder, out_folder):
    # Folder containing images
    folder_path = in_folder
    
    # Output folder for converted images (optional: set to folder_path to overwrite)
    output_folder = out_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
    
        try:
            # Open the image
            with Image.open(file_path) as img:
                # Check image mode (bit depth)
                if img.mode in ["RGB", "L"]:  # RGB (8-bit color), L (8-bit grayscale)
                    print(f"✅ {filename}: Already 8-bit")
                    img.save(os.path.join(output_folder, filename))  # Save as is
                else:
                    print(f"🔄 {filename}: Converting to 8-bit RGB...")
                    img = img.convert("RGB")  # Convert to 8-bit RGB
                    img.save(os.path.join(output_folder, filename))
    
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    print("✅ All images checked and converted if needed!")
    
def training_model(model, train_gen, val_gen, calls, steps_t, steps_v, eps=100):
    model.fit(
        x=train_gen,
        y=None,
        batch_size=None,
        epochs=eps,
        verbose='auto',
        callbacks=calls,
        validation_split=0.0,
        validation_data=val_gen,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=steps_t,
        validation_steps=steps_v,
        validation_batch_size=None,
        validation_freq=1
    )
    return model

    

def transfer_learn_model(model, train_gen, val_gen, calls, classes=75, eps=100, IsTrainable=False):
    # Load pretrained without the classification head
    #removed top classification layer from the model and replace
    #with my own globalavgpooling, dense layer and softmax for correct no of classes
    #it is better to add some pooling and dense layers before the final softmax, 
    #should give better performance
    base_model = model
    
    # Freeze the base model
    base_model.trainable = IsTrainable
    base_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    for layer in base_model.layers:
        print(layer.name, "Trainable:", layer.trainable)
    # Add a custom classifier
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation="relu"), 
        keras.layers.Dense(classes, activation="softmax")  # Adjust for your classes
    ])
    
    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    for layer in model.layers:
        print(layer.name, "Trainable:", layer.trainable)
    # Train on your dataset passed as parameter to the function
    model.fit(train_gen,
        steps_per_epoch=len(train_gen),
        epochs=eps,
        callbacks=calls,
        validation_data=val_gen,
        validation_steps=len(val_gen))
    
    return model

def resume_training(x_train, x_val, suffix, epochs=20, pat=6, mindel=0.005, start_from_epoch=0):
    cl="./files/"+suffix+"/LastCallback.keras"
    cs="./files/"+suffix+"/LastCallbackResume.keras"
    callback_store = cs
    checkpoint = ModelCheckpoint(callback_store, monitor='val_accuracy', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_accuracy', restore_best_weights=True, min_delta=mindel, patience=pat, mode='auto')
    csv_logger = CSVLogger("./files/"+suffix+"/training_res_log.csv", append=True)
    callback_list = [checkpoint, earlystop, csv_logger]
    
    model = keras.models.load_model(cl)
    model.fit(
        x=x_train,
        epochs=epochs,
        verbose="auto",
        callbacks=callback_list,
        validation_split=0.0,
        validation_data=x_val,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=start_from_epoch,
        steps_per_epoch=len(x_train),
        validation_steps=len(x_val),
        validation_freq=1,
    )
    
    
    return model

def plot_training_history(model, suffix):
    # Extract accuracy and validation accuracy
    #pass the trained model
    history = model
    # Extract metrics
    epochs = range(1, len(history.history["accuracy"]) + 1)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot accuracy on the first Y-axis (left)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.plot(epochs, acc, "bo-", label="Training Accuracy")  # "bo-" = blue circles
    ax1.plot(epochs, val_acc, "bs-", label="Validation Accuracy")  # "bs-" = blue squares
    ax1.tick_params(axis="y", labelcolor="blue")
    
    # Create second Y-axis (right) for loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="red")
    ax2.plot(epochs, loss, "ro--", label="Training Loss")  # "ro--" = red circles, dashed
    ax2.plot(epochs, val_loss, "rs--", label="Validation Loss")  # "rs--" = red squares, dashed
    ax2.tick_params(axis="y", labelcolor="red")
    
    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
    
    # Title
    plt.title("Training vs Validation Accuracy & Loss")
    
    plt.savefig("./files/"+suffix+"/training_log.png", format="png", dpi=300, bbox_inches="tight")  # High-res PNG
    #plt.show()
    


def rebuild_model(x_train, x_val, suffix, drop=0.2, classes=75, epochs=10):
    
    callback_store = "./files/"+suffix+"/LastCallbackRebuild.keras"
    checkpoint = ModelCheckpoint(callback_store, monitor='val_accuracy', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_accuracy', restore_best_weights=True, min_delta=0.001, patience=5, mode='auto')
    
    callback_list = [checkpoint, earlystop]
    
    # Load the saved model
    old_model = keras.models.load_model("./files/"+suffix+"/LastCallback.keras")
    #
    #print(type(old_model)) - checking if model is sequential
    #print(old_model.summary())
    # Define a new input layer with the correct shape
    # Manually define an input layer (since old_model.input is undefined)
    new_input = Input(shape=(224, 224, 3))  # Match original input shape

    # Pass the input through all layers EXCEPT the last (removing softmax)
    x = new_input
    for layer in old_model.layers[:-1]:  # Skip last layer
        x = layer(x)  # Pass input through each layer
    
    
    # Add new layers on top
    x = Dropout(drop)(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(classes, activation="softmax")(x)  # Adjust number of classes

    # Create a new functional model
    new_model = keras.Model(inputs=new_input, outputs=x)
    
    #freeze existing layers
    for layer in new_model.layers[:-3]:
        layer.trainable=False
    
    # Compile new model
    new_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Continue training
    new_model.fit(x_train, validation_data=x_val, epochs=epochs, 
                  callbacks=callback_list, steps_per_epoch=len(x_train), 
                  validation_steps=len(x_val))
    
    return new_model


def plot_and_save_essentials(all_df, mis_df, suffix, get_cmap='Greens'):
    my_map = get_cmap
    my_map_r = get_cmap+"_r"
    mislabel_df = mis_df
    testing_all=all_df
    
    #save results and labels
    all_df.to_csv("./files/"+suffix+"/results.csv")
    class_dict = all_df.set_index("y_true")["y_true_label"].to_dict()
    class_df = pd.DataFrame(list(class_dict.items()), columns=["y_true", "y_true_label"])
    class_df.to_csv("class_mapping.csv", index=False)
    # Save the dictionary to a JSON file
    with open("./files/"+suffix+"/class_mapping.json", "w") as f:
        json.dump(class_dict, f)
    #from here is some code to plot some graphs
    #which should help visualize which classes were difficult to predict correctly
    #
    # Count misclassifications per true label - barchart
    misclass_counts = mislabel_df["y_true_label"].value_counts()

    plt.figure(figsize=(16, 8))
    sns.barplot(x=misclass_counts.index, y=misclass_counts.values, palette=my_map_r, hue=misclass_counts.index)
    plt.xticks(rotation=90)
    plt.xlabel("True Label")
    plt.ylabel("Misclassification Count")
    plt.title("Most Misclassified Classes")
    plt.tight_layout()
    plt.savefig("./files/"+suffix+"/MisClassCount.png", format="png", dpi=300, bbox_inches="tight")
    #plt.show()



    #misclas heatmap - unstack converts groupby object na matrix df
    heatmap_df = mislabel_df.groupby(['y_true_label', 'y_pred_label']).size().unstack(fill_value=0)

    #grouped_df = mislabel_df.groupby(['y_true_label', 'y_pred_label']).size().sort_values(ascending=False)
    grouped_df = mislabel_df.groupby(['y_true_label', 'y_pred_label']).size().reset_index(name='Freq').sort_values('Freq', ascending=False)
    grouped_df.to_csv("./files/"+suffix+"/MisclasPairs.csv")

    plt.figure(figsize=(24, 24))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap=my_map)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Misclassification Heatmap")
    plt.savefig("./files/"+suffix+"/MisClassHeat.png", format="png", dpi=300, bbox_inches="tight")
    plt.savefig("./files/"+suffix+"/MisClassHeat.svg", format="svg")
    #plt.show()



    #plotting classes with lowest recall - compute recall on each class - average micro parameter is doing that
    class_recall = testing_all.groupby("y_true_label").apply(
        lambda x: recall_score(x["y_true"], x["y_pred"], average="micro")
    )
    # Sort recall values in ascending order - otherwise it is up and down
    class_recall_sorted = class_recall.sort_values(ascending=True)


    plt.figure(figsize=(16, 8))
    sns.barplot(x=class_recall_sorted.index, y=class_recall_sorted.values, palette=my_map_r, hue=class_recall_sorted.index)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.title("Per-Class Recall (Hardest Classes on the Left)")
    plt.tight_layout()
    plt.savefig("./files/"+suffix+"/PerClassRecall.png", format="png", dpi=300, bbox_inches="tight")
    plt.savefig("./files/"+suffix+"/PerClassRecall.svg", format="svg")
    #plt.show()
    
def create_compile_re_se_model_02(leaky='relu', classes=75):
    def bottleneck_residual_block(x, filters, reduction=4):
        reduced_filters = filters // reduction  # Reduce computation
        res = Conv2D(reduced_filters, (1, 1), padding='same', activation=leaky)(x)  # 1x1 Conv
        res = BatchNormalization()(res)
        
        res = Conv2D(reduced_filters, (3, 3), padding='same', activation=leaky)(res)  # 3x3 Conv
        res = BatchNormalization()(res)
        
        res = Conv2D(filters, (1, 1), padding='same')(res)  # Restore original shape
        res = BatchNormalization()(res)
        
        res = Add()([x, res])  # Skip connection
        return Activation(leaky)(res)  

    # Squeeze-and-Excitation Block (Lighter Version)
    def se_block(input_tensor, reduction=16):
        filters = input_tensor.shape[-1]  
        se = GlobalAveragePooling2D()(input_tensor)  
        se = Dense(filters // reduction, activation='relu')(se)  
        se = Dense(filters, activation='sigmoid')(se)  
        return Multiply()([input_tensor, se])  

    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    #x = GaussianNoise(stddev=0.03)(input_layer),
    # Initial Conv Layer
    x = Conv2D(128, (5, 5), padding='same', activation=leaky)(input_layer)
    x = BatchNormalization()(x)

    # Bottleneck Residual + SE Block 1
    x = bottleneck_residual_block(x, 128)
    x = se_block(x)
    
    # Bottleneck Residual + SE Block 2
    x = Conv2D(128, (5, 5), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 128)
    x = se_block(x)
    
    # Bottleneck Residual + SE Block 2
    x = Conv2D(128, (3, 3), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 128)
    x = se_block(x)

    # Bottleneck Residual + SE Block 3
    x = Conv2D(256, (3, 3), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 256)
    x = se_block(x)
    x = Dropout(0.1)(x)

    # Bottleneck Residual + SE Block 4
    x = Conv2D(512, (3, 3), padding='same', strides=2, activation=leaky)(x)  # Downsampling
    x = BatchNormalization()(x)
    x = bottleneck_residual_block(x, 512)
    x = se_block(x)

    # Global Average Pooling - instead of Flatten
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = Dense(512, activation=leaky)(x)  # Reduced size
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation=leaky)(x)  # Reduced size
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    # Output Layer
    output_layer = Dense(classes, activation='softmax')(x)

    # Create model
    model_CNN = Model(inputs=input_layer, outputs=output_layer)

    # Compile Model - lower learning rate
    model_CNN.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    return model_CNN