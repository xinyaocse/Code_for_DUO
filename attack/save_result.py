from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.cm as cm

def visulization(self):
    for k in range(self.query.shape[0]):
        s_path = self.save_path + '/' + self.query_path[k].split('/')[-1].split('.')[0] + '_target_' \
                 + self.target_path[k].split('/')[-1].split('.')[0]
        show_video(self.unnormalization(self.query[k]), s_path + '_ori.pdf')
        show_video(self.unnormalization(self.query[k] + self.theta[k] * self.mask_I[k] * self.mask_F[k]),
                   s_path + '_adv.pdf')
        show_video(self.theta[k] * self.mask_I[k] * self.mask_F[k] * 255, s_path + '_perturb.pdf')
        show_pertub_hotFig((self.theta[k] * self.mask_I[k] * self.mask_F[k]).detach().cpu().numpy(),
                           s_path + '_hot.pdf')


def show_video(self, data, save_path):
    imshow_imgFrame = []
    for length in range(16):
        img_frame = data[length].cpu().permute(1, 2, 0).numpy().astype('uint8')
        imshow_imgFrame.append(img_frame)

    with PdfPages(save_path) as pdf:
        plt.figure()  #
        plt.rcParams['figure.dpi'] = 800
        for i in range(4):
            for j in range(4):
                plt.subplot(4, 4, i * 4 + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.imshow(imshow_imgFrame[i * 4 + j])
        pdf.savefig()
        plt.close()


def show_pertub_hotFig(self, data, save_path):
    imshow_imgFrame = []
    for length in range(16):
        hotdata = data[length]
        hotdata = hotdata.mean(0)
        hotdata = np.maximum(hotdata, 0)
        hotdata = hotdata - np.min(hotdata)
        hotdata = hotdata / np.max(hotdata)
        hotdata = hotdata
        imshow_imgFrame.append(hotdata)

    with PdfPages(save_path) as pdf:
        plt.figure()  #
        plt.rcParams['figure.dpi'] = 800
        for i in range(4):
            for j in range(4):
                plt.subplot(4, 4, i * 4 + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.imshow(imshow_imgFrame[i * 4 + j], extent=(0, 224, 0, 224), cmap=cm.hot)
                plt.colorbar()
        pdf.savefig()
        plt.close()

def save_obj(self, data, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_obj(self, path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)