# import zipfile
import zipfile
def extract_zip_file(filename):
    '''
    This functions unzips the zip file whose location is provided as an argument.
    '''
    zipfile_ref=zipfile.ZipFile(filename,'r')
    zipfile_ref.extractall()
    zipfile_ref.close()
    

import os   
def walk_through_dir(directory_name):
    
    '''
    Accepts the dirname as argument and prints the contents of each directory sequentially.
    It prints the sub-directories and number of images present in each.
    '''
    for dirpaths,dirnames,filenames in os.walk(directory_name):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpaths}'")

        
        
        
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_random_image(target_dir):
    """
    takes the directory as input and prints 5 random images from the randomly choosen class.
    """
    target_class=random.choice(os.listdir(target_dir))
    target_folder=os.path.join(target_dir,target_class)
    random_image=random.sample(os.listdir(target_folder),5)
 
    plt.figure(figsize=(16,5))
    for i in range(5):
        
        plt.subplot(1,5,i+1)
        img=tf.io.read_file(os.path.join(target_folder,random_image[i]))
        img=tf.io.decode_image(img)
        plt.imshow(img)
        plt.title(f'{target_class}\n{img.shape}')
        plt.axis(False) 
        
import matplotlib.pyplot as plt
# ploting training curve seperately
def plot_loss_curves(history):
    
    '''
      returns seperate loss curves for training and validation metrics
    '''
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']

    train_accuracy=history.history['accuracy']
    val_accuracy=history.history['val_accuracy']

    epochs=range(1,len(history.history['loss'])+1)
    plt.figure(figsize=(20,7))
  # plot loss data
    plt.subplot(1,2,1)
    plt.plot(epochs,train_loss,label="training_loss")
    plt.plot(epochs,val_loss,label="validation_loss")
    plt.title("Loss curves")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
  # plt.show()

  # plot accuracy data
    plt.subplot(1,2,2)
    plt.plot(epochs,train_accuracy,label="training_acc")
    plt.plot(epochs,val_accuracy,label="validation_acc")
    plt.title("Accuracy curves")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()



import tensorflow as tf
import datetime
def create_tensorboard_callback(dir_name,experiment_name):
    """
    creates a tensorBoard callback and returns the created callback obj.
    args:
    dir_name:name of the directory to store the logs.
    experiment_name:name of the experiment.
    """
    log_dir=os.path.join(dir_name,experiment_name,str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    tensorflow_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving tensorboard log files to '{log_dir}'.")
    return tensorflow_callback

import matplotlib.pyplot as plt
import random
import os


def plot_original_and_augmented(train_dir,class_names,data_augmentation_layer):
    """
    accepts the directory,class_names and data augmentation layer 
    and prints a random image from the random class with and without 
    data augmentation.
    """
    target_class=random.choice(class_names)
    target_dir=os.path.join(train_dir,target_class)
    random_image=random.choice(os.listdir(target_dir))
    img=tf.io.read_file(os.path.join(target_dir,random_image))
    img=tf.io.decode_image(img)
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis(False)
    plt.title(f'Original Image from class: {target_class}');
    # plot the augmented image
    augmented_image=data_augmentation_layer(img)
    plt.subplot(1,2,2)
    plt.imshow(augmented_image/255.0)
    plt.axis(False)
    plt.title(f'Augmented Image from class: {target_class}');


    
def plot_and_compare_history(original_history,new_history,initial_epoch):
    """
    the function accepts the histories of a model before and after fine-tunning.
    initial_epoch:#epochs used to train the original model.
    """
    #get original history measurements
    acc=original_history.history['accuracy']
    loss=original_history.history['loss']
    val_acc=original_history.history['val_accuracy']
    val_loss=original_history.history['val_loss']
    
    #combining 
    total_acc=acc+new_history.history['accuracy']
    total_loss=loss+new_history.history['loss']
    total_val_acc=val_acc+new_history.history['val_accuracy']
    total_val_loss=val_loss+new_history.history['val_loss']
    
    #make plots
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(total_acc,label='Training Accuracy')    
    plt.plot(total_val_acc,label='Validation Accuracy')
    plt.plot([initial_epoch-1,initial_epoch-1],plt.ylim(),label='Start fine tunning')
    plt.title("Accuracy")
    plt.legend(loc='lower right')
    plt.subplot(1,2,2)
    plt.plot(total_loss,label='Training loss')
    plt.plot(total_val_loss,label="Validation loss")
    plt.plot([initial_epoch-1,initial_epoch-1],plt.ylim(),label='Start fine tunning')
    plt.title("Loss")
    plt.legend(loc="upper right")
    
    
    
    
import tensorflow as tf
def create_model_check_point_callback(checkpoint_path,monitor='val_loss'):
    """
    Takes the path where to save the best model weights obtained during training.
    """
    model_checkpoint_cb=tf.keras.callbacks.ModelCheckpoint(
        
        monitor=monitor,
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    return model_checkpoint_cb




from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def make_confusion_matrix(y_true,y_preds,class_names):
    
    cm=confusion_matrix(y_true,y_preds)
    # cm_norm=cm.astype('float')/cm.sum(axis=1)
    # cm_norm=cm.astype('float')
  # print(cm.sum(axis=1).shape)
  # print(cm.sum(axis=1)[:np.newaxis].shape)
    plt.figure(figsize=(100,100))
    sns.heatmap(cm,annot=True,cmap='Blues',fmt='.0f')
    plt.title('Confusion Matrix')
    plt.ylabel("True values")
    plt.xlabel('Predicted values')
    plt.xticks(ticks=np.arange(len(class_names))+0.5,labels=class_names,rotation=90)
    plt.yticks(ticks=np.arange(len(class_names))+0.5,labels=class_names,rotation=0)

    
import tensorflow as tf    
def load_and_prep_image(filename,img_shape=224,scale=True):
    
    '''
    reads an image from the filename and turns it into a tensor,
    and reshapes it to the specified size.
    
    args:
    filename(str):path to the target image.
    img_shahape(a,b,c)=target shape.
    scale(boolean): specifies wheather scaling is required to be done or not.
    
    returns:
    image tensor with the target shape.
    '''
    
    img=tf.io.read_file(filename)
    img=tf.io.decode_image(img,channels=3)
    img=tf.image.resize(img,size=[img_shape,img_shape])
    
    if scale:
        img=img/255.0
    return img



"""
Data Augmentation layer code:
data_augmentation=keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomRotation(0.2,fill_mode='nearest'),
    preprocessing.RandomZoom(0.2),
    # preprocessing.Rescaling(scale=1.0/255)
],name='Data_Augmentation_Layer')
"""

            

'''
to split input data into train and test
input_data_dir='/kaggle/input/formula-one-cars/Formula One Cars'
for folder in os.listdir(input_data_dir):
    files=os.listdir(os.path.join(input_data_dir,folder))
    images=[]
    for f in files:
        try:
            img=tf.io.read_file(os.path.join(input_data_dir,folder,f))
            img=tf.image.decode_image(img)
            if img.ndim == 3:
                images.append(f)
        except:
               pass
            
            
    random.shuffle(images)
    count=len(images)
    split=int(0.8*count)
    os.mkdir(os.path.join('./data/train',folder))
    os.mkdir(os.path.join('./data/test',folder))

    for c in range(split):
        source_file=os.path.join(input_data_dir,folder,images[c])
        distination=os.path.join('./data/train',folder,images[c])
        copyfile(source_file,distination)
    for c in range(split,count):
        source_file=os.path.join(input_data_dir,folder,images[c])
        distination=os.path.join('./data/test',folder,images[c])
        copyfile(source_file,distination)
    '''
