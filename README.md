# Speech enhancement with spiking neural network
Denoise Speech by a Spiking Neural Network converted from an Artificial Neural Network (Using NengoDL, Keras and Tensorflow) 

------------------

This work is part of a research project at the engineering faculty of the University of Sherbrooke. This project is carried out by Idrissi Ismail (CIP: idri3201) and supervised by Professor Éric Plourde.
This project is modified from ClearWave (DNN) by Boozyguo (https://github.com/boozyguo/ClearWave). 

Also, the project uses ffmpeg and pesq to deal with speech data.

Before try the project, please download the base dnn model from https://drive.google.com/file/d/1jQjI5IkFq1imAcvAzK_gnE_Q983N9unJ/view?usp=sharing
 and copy the .h5 file to ./models/pretrained/base_dnn_model.h5.


------------------
Preparation of the execution environment:

0. The code for this project runs on a Linux system. If you rather have a Windows system, you can follow the necessary instructions on this link https://docs.microsoft.com/en-us/windows/wsl/install-win10 to install a Linux distribution with WSL (Windows Subsystem for Linux ) then continue with the following steps.
1. Install python. To do this, execute in order the following commands:
```
 apt list --upgradable
 sudo apt update
 sudo apt install software-properties-common
 sudo add-apt-repository ppa:deadsnakes/ppa
 sudo apt install python3.8
```
2. Install the necessary packages using the requirements.txt file. To do this, access the project folder then execute the following commands:
```
pip install -r requirements.txt OR pip3 install -r requirements.txt	
pip install keras
apt install ffmpeg
```
3. In order to run the code, you must have permission to run the project files. To do this, run the command:
```
chmod -R 755 <directory_name>
```
4. To avoid some coding errors that may occur, we execute the following commands to remove trailing \ r character:
```
sed -i 's/\r$//' demo.sh
sed -i 's/\r$//' runme.sh
```

Now your environment is ready to enhance speech!

------------------

## Inference :
To use the inference with ANN, change the parameter predict_using_snn in config.py file to the value False. 
By default, the inference is done by the SNN. 
The procedure for running the inference code to improve your speech is :

------------------

### Inference Usage: Denoise on noisy data. 
If you have noisy speech, you can edit and run "./demo.sh" to denoise the noisy file. 

1. Put the noisy file in path "./demo_data/noisy/*.wav"
2. Edit the demo.sh file with "INPUT_NOISY=1" 
3. Run ./demo.sh
4. Check the denoised speech in "demo_workspace/ns_enh_wavs/test/1000db/*.wav"

### Inference Usage: Denoise on speech data and noise data. 
If you have clear speech and noise: 

1. Put the noise file in path "./demo_data/noise/*.wav"
2. Put the clear speech file in path "./demo_data/clear/*.wav"
3. Edit the demo.sh file with "INPUT_NOISY=0". Also, you can modify the SNR in parameter "TE_SNR", for example "TE_SNR=5" is 5db.
4. Run ./demo.sh
5. Check the denoised speech in "demo_workspace/ns_enh_wavs/test/5db/*.wav" (if "TE_SNR=5") 

------------------

## Training Usage: Training model on speech data and noise data. 
If you want to train yourself model, just prepare your data, then run "./runme.sh".
Training the model using noise and clean data allows you to calculate and store the weights of your model, which you can use later for enhancing speech with the spiking neural network.
The procedure for calculating and storing the weights of your model is as follows:

1. Put the train noise file in path "./data/train_noise/*.wav"
2. Put the train clear speech file in path "./data/train_speech/*.wav"
3. Put the validtaion noise file in path "./data/test_noise/*.wav"
4. Put the train validtaion speech file in path "./data/test_speech/*.wav"
5. Edit the runme.sh file, set parameters: TR_SNR, TE_SNR, EPOCHS, LEARNING_RATE
6. Run ./runme.sh
7. Check the new model in "./workspace/models/5db/*.h5" (if "TE_SNR=5") 

------------------


## Models:

ClearWave model based on simple DNN in keras:

```python
    n_concat = 7
    n_freq = 257
    n_hid = 2048
    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dropout(0.1))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='relu'))
    model.summary()
```


------------------

## Ref:

 https://github.com/boozyguo/ClearWave
 
