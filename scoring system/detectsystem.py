import pylab as plt
import phcx
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ultralytics import YOLO
from scipy import stats
import matplotlib.patches as patches


class PulsarDetection:
    def __init__(self, input_path):
        self.input_path = input_path

        if os.path.isfile(self.input_path):
            self.files = [self.input_path]
            self.output_dir = os.path.dirname(self.input_path)
        elif os.path.isdir(self.input_path):
            self.files = [os.path.join(self.input_path, f) for f in os.listdir(self.input_path) if f.endswith('.phcx')]
            self.output_dir = os.path.join(os.path.dirname(self.input_path), os.path.basename(self.input_path) + '_PulsarCandScore')
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            raise ValueError("输入路径既不是文件也不是文件夹")

        # laod model
        self.modelOD = YOLO('HTRU2ph.pt')
        self.modelscore = load_model('HTRUrank.h5')

    def phase_plots(self, cand, fname, bandpre, intpre, conf):
        fig = plt.figure(dpi=300)

        ax1 = plt.subplot2grid((3, 2), (0, 0))
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax3 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
        ax4 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)

        ax1.axis('off')
        max_index = np.argmax(cand.profile)
        array = np.roll(cand.profile, len(cand.profile) // 2 - max_index)
        arrayprofile = np.concatenate((array, array))

        ax1.plot(range(cand.profile.size * 2), arrayprofile, c='k')
        ax1.set_title('Profile')

        ax2.plot(cand.dm_values, cand.snr_values, c='k')
        ax2.set_xlabel('DM (pc/cm$^3$)')
        ax2.set_ylabel('SNR')
        ax2.set_title('DM-SNR')

        # plot Sub-Bands (ax3)
        array = np.roll(cand.subbands, cand.subbands.shape[1] // 2 - max_index, axis=1)
        arrayband = np.concatenate((array, array), axis=1)
        ax3.imshow(arrayband, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
        ax3.set_title('Sub-Bands')
        ax3.set_xlabel('Bin Index')
        ax3.set_ylabel('Band Index')

        num_xticks = 5  # x label 
        x_labels = [0, 0.5, 1, 1.5, 2.0]
        xticks = np.linspace(0, arrayband.shape[1] - 1, num_xticks)
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(x_labels)

        # plot Sub-Integrations (ax4)
        arrayint = np.roll(cand.subints, cand.subints.shape[1] // 2 - max_index, axis=1)
        arrayint2 = np.concatenate((arrayint, arrayint), axis=1)
        ax4.imshow(arrayint2, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
        ax4.set_title('Sub-Integrations')
        ax4.set_xlabel('Bin Index')
        ax4.set_ylabel('Integration Index')

        # plot outlays
        self._draw_boxes(ax3, bandpre, arrayband.shape)
        self._draw_boxes(ax4, intpre, arrayint2.shape)

        plt.tight_layout()
        fname = str(conf[0])+fname
        save_path = os.path.join(self.output_dir, fname)
        plt.savefig(save_path)
        plt.show()

    def _draw_boxes(self, ax, boxes, shape):
        height, width = shape
        for bbox in boxes:
            x_center, y_center, bw, bh, confidence = bbox
            x_center = x_center * width
            y_center = y_center * height
            bw = bw * width
            bh = bh * height
            x_left = x_center - bw / 2
            y_bottom = y_center - bh / 2 - 0.4
            rect = patches.Rectangle((x_left, y_bottom), bw, bh, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_left + bw, y_bottom + bh - 0.4, f'{confidence:.2f}', color='red', fontsize=8, verticalalignment='top')

    def process_file(self, file_path):
        #load data
        cand = phcx.Candidate(file_path)

        max_index = np.argmax(cand.profile)

        # Sub-bands pred
        bandpredict = self._predict_band_or_int(cand.subbands, max_index, 'subbands')

        # Sub-integrations pred
        intpredict = self._predict_band_or_int(cand.subints, max_index, 'subints')

        # DM value
        DMs = self._calculate_dm_stats(cand)

        # data padding
        new_OD_data_padded = pad_sequences([bandpredict], padding='post', dtype='float32', maxlen=bandpredict.shape[1])
        new_INT_data_padded = pad_sequences([intpredict], padding='post', dtype='float32', maxlen=intpredict.shape[1])
        DMdata_new_padded = pad_sequences([DMs], maxlen=4, dtype='float32', padding='post')

        # predect
        predictions = self.modelscore.predict([new_OD_data_padded, new_INT_data_padded, DMdata_new_padded])

        # output
        print(f"Predictions for {os.path.basename(file_path)}: {predictions}")

        # plots
        output_filename = os.path.basename(file_path).replace('.phcx', '.png')
        self.phase_plots(cand, output_filename, bandpredict, intpredict, predictions)

    def _predict_band_or_int(self, array, max_index, array_type):
        array = np.roll(array, array.shape[1] // 2 - max_index, axis=1)
        arrayband = np.concatenate((array, array), axis=1)
        fig = os.path.join(self.output_dir, f'temp_{array_type}.png')
        plt.imshow(arrayband, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.savefig(fig)

        # pred
        predictions = self.modelOD.predict(fig, conf=0.5)
        os.remove(fig)

        pred_list = []
        for p in predictions:
            b1 = p.boxes.xywhn.numpy()
            b2 = p.boxes.conf.numpy()
            b2 = b2[:, np.newaxis]
            pred_list.append(np.hstack((b1, b2)))

        return np.vstack(pred_list)

    def _calculate_dm_stats(self, cand):
        DM = np.array(cand.dm_values)
        DMs = np.array([np.nan_to_num(DM.mean(), nan=0.0),
                        np.nan_to_num(DM.std(), nan=0.0),
                        np.nan_to_num(stats.skew(DM, bias=False), nan=0.0),
                        np.nan_to_num(stats.kurtosis(DM, bias=False), nan=0.0)])
        return DMs

    def process(self):
        for file in self.files:
            self.process_file(file)


if __name__ == "__main__":
    input_path = 'phcxdata' 
    try:
        detector = PulsarDetection(input_path)
        detector.process()
    except ValueError as e:
        print(e)
