########## Flowers Dataset
#Divison of Data is as follows->
#Testing contains 100 images
#Validation contains 200 images
#Rest of the images are in Training

import os,shutil
from keras import layers,models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

original_dir=r'C:\Users\Anu\Downloads\datasets\flowers-recognition\flowers'
base_dir=r'C:\Users\Anu\Downloads\datasets\flowers-recognition\model_data'
if(os.path.isdir(base_dir)):
    print('Base Dir Exist')
else:
    os.mkdir(base_dir)

# Train Dir
train_dir=os.path.join(base_dir,'train')
if(os.path.isdir(train_dir)):
    print('Directory Train Exist')
else:
    os.mkdir(train_dir)

#Validation Dir
validation_dir=os.path.join(base_dir,'validation')
if(os.path.isdir(validation_dir)):
    print('Directory Validation Exist')
else:
    os.mkdir(validation_dir)

#Test Dir
test_dir=os.path.join(base_dir,'test')
if(os.path.isdir(test_dir)):
    print('Directory test Exist')
else:
    os.mkdir(test_dir)


#List all the files in Original Dir
Labels_list=os.listdir(original_dir)
for label in Labels_list:
    files_list=os.listdir(os.path.join(original_dir,label))
    total_files=len(files_list)
    #Validation
    fnames_validation=files_list[0:200]
    for fname in fnames_validation:
        source_dir=os.path.join(original_dir,label)
        destination_dir=os.path.join(validation_dir,label)
        if ((os.path.isdir(destination_dir)==True)):
            src=os.path.join(source_dir,fname)  
            des=os.path.join(destination_dir,fname)
            shutil.copyfile(src,des)
        else:
            os.mkdir(destination_dir)
            src=os.path.join(source_dir,fname)  
            des=os.path.join(destination_dir,fname)
            shutil.copyfile(src,des)
     #Testing   
    fnames_testing=files_list[200:300]
    for fname in fnames_testing:
        source_dir=os.path.join(original_dir,label)
        destination_dir=os.path.join(test_dir,label)
        if ((os.path.isdir(destination_dir)==True)):
            src=os.path.join(source_dir,fname)  
            des=os.path.join(destination_dir,fname)
            shutil.copyfile(src,des)
        else:
            os.mkdir(destination_dir)
            src=os.path.join(source_dir,fname)  
            des=os.path.join(destination_dir,fname)
            shutil.copyfile(src,des)
     #Training   
    fnames_training=files_list[300:len(files_list)]
    for fname in fnames_training:
        source_dir=os.path.join(original_dir,label)
        destination_dir=os.path.join(train_dir,label)
        if ((os.path.isdir(destination_dir)==True)):
            src=os.path.join(source_dir,fname)  
            des=os.path.join(destination_dir,fname)
            shutil.copyfile(src,des)
        else:
            os.mkdir(destination_dir)
            src=os.path.join(source_dir,fname)  
            des=os.path.join(destination_dir,fname)
            shutil.copyfile(src,des)


########## Model Creation
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu',padding='same',use_bias=True))
model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same',use_bias=True))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',use_bias=True))
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',use_bias=True))
model.add(layers.Dropout(rate=0.4))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same',use_bias=True))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same',use_bias=True))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(5,activation='sigmoid'))

##### Data Preparetion
image_gen=ImageDataGenerator(rescale=1./255,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             zoom_range=0.2,
                             shear_range=0.2,
                             rotation_range=40)
train_data=image_gen.flow_from_directory(train_dir,
                                         target_size=(150,150), 
                                         batch_size=30,
                                         class_mode='categorical',
                                         shuffle=False)

val_data=image_gen.flow_from_directory(validation_dir,
                                         target_size=(150,150),
                                         batch_size=30,
                                         class_mode='categorical',
                                         shuffle=False)

test_image_gen=ImageDataGenerator(rescale=1./255)
test_data=test_image_gen.flow_from_directory(test_dir,
                                             target_size=(150,150))


### Optimization
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit_generator(train_data,
                            steps_per_epoch=100,
                            epochs=30,
                            validation_data=val_data,
                            validation_steps=100)
model.save('Flower_Classification_keras.h5')



acc=history.history['acc']
loss=history.history['loss']
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'b',color='red',label='Training Accuracy')
plt.plot(epochs,val_acc,'b',color='blue',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'b',color='red',label='Training Loss')
plt.plot(epochs,val_loss,'b',color='blue',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.figure()
plt.show()


######### Load saved model
from keras.models import load_model

model=load_model(r'C:\Users\Anu\Downloads\datasets\flowers-recognition\Flower_Classification_keras.h5')
model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit_generator(test_data,
                            steps_per_epoch=100,
                            epochs=30)
### Metric Plot
acc=history.history['acc']
loss=history.history['loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'b',color='red',label='Test Accuracy')
plt.title('Testing Accuracy')
plt.figure()
plt.plot(epochs,loss,'b',color='blue',label='Testing Loss')
plt.title('Testing Loss')
plt.legend()
plt.figure()
plt.show()
                          