"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong & Idrissi Ismail
Created:  2017.12.18
Modified: 2021.05.18  (adding spiking neural network code, and some small changes)

"The code of the spiking neural network is added by Idrissi Ismail and supervised by Prof. Éric Plourde"

"""
import numpy as np
import random as random
import os
import pickle
import pickle as cPickle
import argparse
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import nengo as nengo
import nengo_dl as nengo_dl
tf.compat.v1.disable_eager_execution()
import prepare_data as pp_data
import config as cfg
from data_generator import DataGenerator
from evaluate import calculate_pesq
from spectrogram_to_wave import recover_wav
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam


def eval(model, gen, x, y):
    """Validation function.

    Args:
      model: keras model.
      gen: object, data generator.
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []

    # Inference in mini batch.
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
        if False:
            print("pred")
            print(pred)

    # Concatenate mini batch prediction.
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Compute loss.
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss


def train(args):
    """Train the neural network. Write out model every several iterations.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      lr: float, learning rate.
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    calc_log = args.calc_log
    epoch = args.epoch

    scale = True

    # Load data.
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,))

    batch_size = 128  # 128 #500
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    ""
    # Scale data.

    t1 = time.time()
    if calc_log:
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr),
                                   "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        tr_y = pp_data.scale_on_2d(tr_y, scaler)
        te_x = pp_data.scale_on_3d(te_x, scaler)
        te_y = pp_data.scale_on_2d(te_y, scaler)
    else:
        print("max of tr_x:", np.max(tr_x))
        print("max of tr_y:", np.max(tr_y))
        print("max of te_x:", np.max(te_x))
        print("max of te_y:", np.max(te_y))
        tr_x = tr_x / np.max(tr_x)
        tr_y = tr_y / np.max(tr_y)
        te_x = te_x / np.max(te_x)
        te_y = te_y / np.max(te_y)

    print("Scale data time: %s s" % (time.time() - t1,))

    # Debug plot.
    if True:
        plt.figure()
        plt.matshow(tr_x[0: 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        print("----------------- PLOOOOT SHOOOOWEEED -----------------")
        # pause

    # Build model
    (_, n_concat, n_freq) = tr_x.shape
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

    if calc_log:
        model.add(Dense(n_freq, activation='linear'))
    else:
        model.add(Dense(n_freq, activation='relu'))
    model.summary()

    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=lr))

    # Data generator.
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)

    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)

    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    pp_data.create_folder(stats_dir)

    # Print loss before training.
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    if False:
        print("tr_x")
        print(tr_x)
        print("tr_y")
        print(tr_y)
        print("te_x")
        print(te_x)
        print("te_y")
        print(te_y)
    print("Iteration: %d, tr_loss: %2.20f, te_loss: %2.20f" % (iter, tr_loss, te_loss))

    # Save out training stats.
    stat_dict = {'iter': iter,
                 'tr_loss': tr_loss,
                 'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    # Train. ./models/pretrained/base_dnn_model.h5
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1

        # Validate and save training stats.
        if iter % 200 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %2.20f, te_loss: %2.20f" % (iter, tr_loss, te_loss))

            # Save out training stats.
            stat_dict = {'iter': iter,
                         'tr_loss': tr_loss,
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        # Save model.
        if iter % 200 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            print(model_path)
            # model.save(model_path)
            model.save_weights(model_path)
            print("Saved model to %s" % model_path)

        if iter == (epoch + 1):
            break

    print("Training time: %s s" % (time.time() - t1,))


def inference(args):
    """Inference all test data, write out recovered wavs to disk.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      n_concat: int, number of frames to concatenta, should equal to n_concat
          in the training stage.
      iter: int, iteration of model to load.
      visualize: bool, plot enhanced spectrogram for debug.
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration
    calc_log = args.calc_log
    model_file = args.model_file

    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True

    # Spiking neural network parameters
    scale_firing_rates = cfg.scale_firing_rates

    # Build model
    n_concat = 7
    n_freq = 257
    n_hid = 2048
    lr = 1e-3

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
    if calc_log:
        model.add(Dense(n_freq, activation='linear'))
    else:
        model.add(Dense(n_freq, activation='relu'))
    model.summary()

    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=lr))

    # Load model.
    if (model_file == "null"):
        model_path = os.path.join("workspace", "models", "%ddb" % int(tr_snr), "md_%diters.h5" % iter)
        print("||||||||||| model_path model_path model_path model_path model_path model_path :", model_path)
        # model = load_model(model_path)
        model.load_weights(model_path)
    else:
        model.load_weights(model_file)
        # model.load_weights("./models/pretrained/base_dnn_model_just_dense_layers.h5")

    # Convert model to spiking neural network -------- in INFERENCE Functiun Section ----------
    converter = nengo_dl.Converter(model,
                                   allow_fallback=True,
                                   swap_activations={tf.keras.activations.relu: nengo.SpikingRectifiedLinear()},
                                   # swap_activations={tf.keras.activations.relu: nengo.RectifiedLinear()},
                                   scale_firing_rates=scale_firing_rates,
                                   # synapse=0.0001
                                   )

    # Load scaler.
    if calc_log:
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))

    # Load test data.
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature.
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)

        # Process data.
        n_pad = (n_concat - 1) / 2
        mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
        if calc_log:
            mixed_x = pp_data.log_sp(mixed_x)
            # speech_x = pp_data.log_sp(speech_x)
        else:
            mixed_x = mixed_x
            # speech_x = speech_x

        # Scale data.
        if calc_log:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
            # speech_x = pp_data.scale_on_2d(speech_x, scaler)
        else:
            mixed_x_max = np.max(mixed_x)
            print("max of tr_x:", mixed_x_max)
            mixed_x = mixed_x / mixed_x_max

            speech_x_max = np.max(speech_x)
            print("max of speech_x:", speech_x_max)
            speech_x = speech_x / speech_x_max

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

        # Add time dimension to our data before predicting
        print("____________________________________________________________________")
        print("mixed_x_3d shape : ", mixed_x_3d.shape)

        # tile data along the time dimension for however many timesteps we need
        n_steps = cfg.n_steps   # number of timesteps for inference
        data = mixed_x_3d
        data = np.tile(data, (1, n_steps, 1))

        # Define the network object
        with converter.net as net:
            # Change network configuration setting : we'll disable some features we
            # don't need in this example, to improve
            nengo_dl.configure_settings(stateful=False, use_loop=True, inference_only=True, trainable=None)

            # Load model.
            if model_file == "null":
                model_path = os.path.join("workspace", "models", "%ddb" % int(tr_snr), "md_%diters.h5" % iter)
                # model = load_model(model_path)
                model.load_weights(model_path)
            else:
                model.load_weights(model_file)

            # probe on our output layer
            probe_output = converter.outputs[converter.model.output]

            # Define the simulator object
            with nengo_dl.Simulator(net, progress_bar=True) as sim:
                # the Converter will copy the parameters from the Keras model, so we don't
                # need to do any further training

                print("----------  data shape : ", data.shape)
                data = data.reshape((-1, n_steps, n_concat * n_freq))
                print("----------  data shape after reshape((-1, n_concat * n_freq)) : ", data.shape)

                # Prediction with SNN
                start_time = time.time()
                data_pred = sim.predict({converter.inputs[converter.model.input]: data})

                print("execution_duration : ", time.time() - start_time)

                # Get the output of the network
                prediction_snn = data_pred[probe_output]
                prediction_snn = np.mean(prediction_snn, axis=1)    # * n_steps / execution_duration

                print("prediction_snn : ", prediction_snn)
                print("prediction_snn shape : ", prediction_snn.shape)
                print("prediction_snn mean : ", np.mean(prediction_snn))

        if False:
            print(mixed_x_3d)
        print("mixed_x_3d shape : ", mixed_x_3d.shape)

        predict_using_snn = cfg.predict_using_snn    # True
        if predict_using_snn:
            pred = prediction_snn
        else:
            pred = model.predict(mixed_x_3d)

        print(cnt, na)

        if False:
            print("pred")
            print(pred)
            print("speech")
            print(speech_x)

        # Inverse scale.
        if calc_log:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            # speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)
        else:
            mixed_x = mixed_x * mixed_x_max
            # speech_x = speech_x * 16384
            pred = pred * mixed_x_max

        # Debug plot.
        if True:    # args.visualize:
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in range(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.show()
            plt.savefig(workspace+'/plot_inference_log_spectrogram' + na + "_" + str(random.randint(1, 101)) + '.png')

        # Recover enhanced wav.
        if calc_log:
            pred_sp = np.exp(pred)
        else:
            # gv = 0.025
            # pred_sp = np.maximum(0,pred - gv)
            pred_sp = pred

        if False:
            pred_sp = mixed_x[3:-3]

        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude
                                                        # Change after spectrogram and IFFT.

        # Write out enhanced wav.
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        # Write out enhanced pcm 8K pcm_s16le.
        out_pcm_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.pcm" % na)
        cmd = ' '.join(["./ffmpeg -y -i ", out_path, " -f s16le -ar 8000 -ac 1 -acodec pcm_s16le ", out_pcm_path])
        os.system(cmd)

        # Write out webrtc-denoised enhanced pcm 8K pcm_s16le.
        ns_out_pcm_path = os.path.join(workspace, "ns_enh_wavs", "test", "%ddb" % int(te_snr), "%s.ns_enh.pcm" % na)
        ns_out_wav_path = os.path.join(workspace, "ns_enh_wavs", "test", "%ddb" % int(te_snr), "%s.ns_enh.wav" % na)
        pp_data.create_folder(os.path.dirname(ns_out_pcm_path))
        cmd = ' '.join(["./ns", out_pcm_path, ns_out_pcm_path])
        os.system(cmd)
        cmd = ' '.join(["./ffmpeg -y -f s16le -ar 8000 -ac 1 -acodec pcm_s16le -i ", ns_out_pcm_path, "  ",ns_out_wav_path])
        os.system(cmd)

        cmd = ' '.join(["rm ", out_pcm_path])
        os.system(cmd)
        cmd = ' '.join(["rm ", ns_out_pcm_path])
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    parser_train.add_argument('--calc_log', type=int, required=True)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--n_steps', type=float, required=False)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--calc_log', type=int, required=True)
    parser_inference.add_argument('--model_file', type=str, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    parser_inference.add_argument('--n_steps', type=float, required=False)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    parser_calculate_pesq.add_argument('--n_steps', type=float, required=False)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    else:
        raise Exception("Error!")
