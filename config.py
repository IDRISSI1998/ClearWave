"""
Summary:  Config file. 
Author:   Qiuqiang Kong & Idrissi Ismail
Created:  2017.12.21
Modified: -
"""

# Use spiking neural network (with NengoDL simulator) to do speech enhancement (we use ANN if False)
predict_using_snn = True

scale_firing_rates = 500000
n_steps = 5

sample_rate = 16000  # 16000
n_window = 512       # windows size for FFT
n_overlap = 256      # overlap of window

