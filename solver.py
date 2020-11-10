import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from src import network
from pathlib2 import Path
from skimage import img_as_ubyte
from src.tools import ProcessBar, CSV_Recorder, ElapsedTimeProcess, to_tensor, normalize, simple_denorm


def to_numpy_img(img):
    img = img_as_ubyte(simple_denorm(img.squeeze(dim=0)).detach().permute(1, 2, 0).cpu().numpy())
    return img


def normalize_and_to_tensor(img):
    img = to_tensor(normalize(img, mean=0.5, std=0.5)).unsqueeze(dim=0).float()
    return img


def to_tensor_only(img):
    img = to_tensor(img).unsqueeze(dim=0).float()
    return img


class Solver(object):
    def __init__(self, opt, dev):
        self.opt = opt
        self.dev = dev
        self._config_setting()
        self._dataloader_setting()
        self._network_setting()
        self._loss_setting()
        self._training_setting()

    def _config_setting(self):
        # make basic directory, log files
        self.exp_path = Path(self.opt['experiment_folder'])
        self.exp_path.mkdir(parents=True, exist_ok=True)

        # csv recorder
        self.recorder = CSV_Recorder()

    def _dataloader_setting(self):
        if self.opt['task'] == 'Inpaint':
            # get image / mask
            img = imread(self.opt['img_dir']) / 255.0
            mask = imread(self.opt['mask_dir']) / 255.0

            # ToTensor
            self.img = normalize_and_to_tensor(img).to(self.dev)
            self.mask = to_tensor_only(mask).to(self.dev)

            # noise
            self.z = torch.randn_like(self.mask)

        elif self.opt['task'] == 'SuperResolution':
            # get low resolution image
            img = imread(self.opt['img_dir']) / 255.0
            self.img = normalize_and_to_tensor(img).to(self.dev)

            # noise
            self.z = torch.randn([1, 1, *self.opt['output_resolution']]).to(self.dev)

            # downsampler
            self.downsampler = torch.nn.functional.interpolate

        else:
            raise ValueError("task {} is not valid.".format(self.opt['task']))

    def _network_setting(self):
        # network
        self.network = network.Network(in_channels=1, out_channels=3, cnum=self.opt['cnum']).to(self.dev)

        # optimizer
        if self.opt['optim'] == 'adam':
            self.optim = optim.Adam(params=self.network.parameters(), lr=float(self.opt['lr']), betas=self.opt['betas'])

    def _loss_setting(self):
        self.criterion = torch.nn.MSELoss()

    def _training_setting(self):
        # process bar setting
        self.Pb = ProcessBar(max_iter=self.opt['iterations'], prefix='Process: ')

        # elapsed time setting
        self.Eta = ElapsedTimeProcess(max_iter=self.opt['iterations'])

    def save_parameter(self):
        state_dict = dict()
        state_dict['network'] = self.network.state_dict()
        state_dict['optimizer'] = self.optim.state_dict()
        torch.save(state_dict, self.opt['parameters'])

    def fit(self):
        if self.opt['task'] == 'Inpaint':
            self.fit_inpaint()
        elif self.opt['task'] == 'SuperResolution':
            self.fit_super_resolution()
        else:
            raise ValueError('task of {} is not valid'.format(self.opt['task']))

    def fit_inpaint(self):
        # masked image save
        masked_img = self.img * (1 - self.mask) + self.mask
        masked_img = to_numpy_img(masked_img)
        imshow(masked_img)
        plt.show()
        imsave(str(self.exp_path / 'masked_img.jpg'), masked_img)

        # training
        self.Eta.start()
        for iteration in range(self.opt['iterations']):
            for n in [x for x in self.network.parameters() if len(x.size()) == 4]:
                n.data += n.data.clone().normal_() * n.data.std() / 50
            self.optim.zero_grad()

            # forward
            out = self.network(self.z)

            # loss
            loss = self.criterion(out * (1-self.mask), self.img * (1-self.mask))

            # backward
            loss.backward()
            self.optim.step()
            self.recorder.write_data('step_{}'.format(iteration), 'loss', loss.item())

            # Process printing
            elapsed_time = self.Eta.end()
            self.Pb.step(other_info='Loss: {:.4f},    ETA: '.format(loss.item()) + elapsed_time)

            # evaluation
            if iteration % self.opt['evaluation_step'] == 0:
                # show image
                img = to_numpy_img(out)
                imshow(img)
                plt.show()

                # save image
                imsave(str(self.exp_path / '{}.jpg'.format(iteration)), img)
            self.Eta.start()

        self.recorder.to_csv(self.exp_path / 'recorder.csv')
        out = self.network(self.z)
        img = to_numpy_img(out)
        imshow(img)
        plt.show()
        imsave(str(self.exp_path / 'result.jpg'), img)

    def fit_super_resolution(self):
        # training
        for iteration in range(self.opt['iterations']):
            self.optim.zero_grad()

            # forward
            out = self.network(self.z)

            # downsample
            downsample_out = self.downsampler(out, size=[self.img.size(2), self.img.size(3)], mode='bilinear', align_corners=True)

            # loss
            loss = self.criterion(downsample_out, self.img)

            # backward
            loss.backward()
            self.optim.step()
            self.recorder.write_data('step_{}'.format(iteration), 'loss', loss.item())

            # Process printing
            elapsed_time = self.Eta.end()
            self.Pb.step(other_info='Loss: {:.4f},    ETA: '.format(loss.item()) + elapsed_time)

            # evaluation
            if iteration % self.opt['evaluation_step'] == 0:
                # show image
                img = to_numpy_img(out)
                imshow(img)
                plt.show()

                # save image
                imsave(str(self.exp_path / '{}.jpg'.format(iteration)), img)
            self.Eta.start()

        self.recorder.to_csv(self.exp_path / 'recorder.csv')
        out = self.network(self.z)
        img = to_numpy_img(out)
        imshow(img)
        plt.show()
        imsave(str(self.exp_path / 'result.jpg'), img)




