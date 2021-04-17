#!/bin/bash

CMD="main_dnn.py"


MODEL_FILE="./models/pretrained/base_dnn_model.h5"
#MODEL_FILE="null"
#MODEL_FILE="./models/pretrained/densenet121_weights_tf_dim_ordering_tf_kernels_changed.h5"
INPUT_NOISY=0 #1

WORKSPACE="./demo_workspace"
mkdir $WORKSPACE
DEMO_SPEECH_DIR="./demo_data/speech"
DEMO_NOISE_DIR="./demo_data/noise"
DEMO_NOISY_DIR="./demo_data/noisy"
echo "Denoise Demo. "


TR_SNR=5
TE_SNR=5
N_CONCAT=7
N_HOP=2
CALC_LOG=0
#EPOCHS=10000
ITERATION=10000
#LEARNING_RATE=1e-3




CALC_DATA=1
if [ $CALC_DATA -eq 1 ]; then

  if [ $INPUT_NOISY -eq 0 ]; then
      # Create mixture csv.
      echo "Go:Create mixture csv. "
      python3 prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$DEMO_SPEECH_DIR --noise_dir=$DEMO_NOISE_DIR --data_type=test  --speechratio=1

      # Calculate mixture features.
      echo "Go:Calculate mixture features. "

      python3 prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$DEMO_SPEECH_DIR --noise_dir=$DEMO_NOISE_DIR --data_type=test --snr=$TE_SNR


      # Calculate PESQ of all noisy speech.
      echo "Calculate PESQ of all noisy speech. "
      python3 evaluate.py calculate_noisy_pesq --workspace=$WORKSPACE --speech_dir=$DEMO_SPEECH_DIR --te_snr=$TE_SNR

      # Calculate noisy overall stats.
      echo "Calculate noisy overall stats. "
      #python3 evaluate.py get_stats
      python3 evaluate.py get_stats --workspace=$WORKSPACE --type="mixed_audio"

  else
      # Calculate noisy features.
      TE_SNR=1000
      echo "Go:Calculate noisy features. "
      python3 prepare_data.py calculate_noisy_features --workspace=$WORKSPACE --noisy_dir=$DEMO_NOISY_DIR --data_type=test --snr=$TE_SNR
  fi

  echo "Data finish!"
  #exit

fi


# Inference, enhanced wavs will be created.
echo "Inference, enhanced wavs will be created. "
CUDA_VISIBLE_DEVICES=0 python3 $CMD inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION --calc_log=$CALC_LOG --model_file=$MODEL_FILE


# Calculate PESQ of all enhanced speech.
echo "Calculate PESQ of all enhanced speech. "
python3 evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$DEMO_SPEECH_DIR --te_snr=$TE_SNR

# Plot_training_stat.
echo "Plot_training_stat. "
python3 evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=$ITERATION --interval_iter=100

# Calculate overall stats.
echo "Calculate overall stats. "
python3 evaluate.py get_stats --workspace=$WORKSPACE --type="enhanced_waves"


#cmd /k