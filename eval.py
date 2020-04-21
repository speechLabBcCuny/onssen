from attrdict import AttrDict
from utils import AverageMeter
from sklearn.cluster import KMeans
import evaluate, librosa
import argparse, data, json, nn, numpy as np, os, time, torch

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


class tester:
    def __init__(self, args):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        self.dataset = args.dataset
        if "resume_from_checkpoint" in args and args.resume_from_checkpoint=="True":
            self.resume_from_checkpoint = True
        else:
             self.resume_from_checkpoint = False
        if args.cuda_option == "True":
            print("GPU mode on...")
            available_device = get_free_gpu()
            print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')

        # build model
        self.output_path = args.output_path+'/%s_%s_%s'%(self.model_name, self.dataset, self.loss_name)
        saved_dict = torch.load(self.output_path+'/final.mdl')
        self.model = self.init_model(args.model_name, args.model_options)
        self.model.load_state_dict(saved_dict["model"])
        print("Loaded the model...")
        self.data_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'tt', args.cuda_option, self.device)


    def init_model(self, model_name, model_options):
        assert model_name is not None, "Model name must be defined!"
        assert model_name in ["dc", "chimera", "chimera++", "phase", "enhance"],\
        "Model name is not supported! Must be one of (dc, chimera, chimera++, phase)"
        if model_name == "dc":
            model = nn.deep_clustering(model_options)
        elif model_name == "chimera" or model_name == "chimera++":
            model = nn.chimera(model_options)
        elif model_name == "phase":
            model = nn.phase_net(model_options)
        elif model_name == "enhance":
            model = nn.enhance(model_options)

        model.to(self.device)
        return model

    def get_est_sig(self, args):
        """
        args:
            feature_mix: batch x frame x frequency
            embedding: batch x frame x frequency x embedding_dim
            stft_r_mix: batch x frame x frequency
            stft_i_mix: batch x frame x frequency
            sig_ref: batch x num_spk x nsample
        return:
            sig_est: batch x num_spk x nsample
        """
        args = [ele.cpu().detach().numpy() for ele in args]
        [feature_mix, embedding, stft_r_mix, stft_i_mix, sig_ref] = args
        stft_mix = stft_r_mix + 1j * stft_i_mix
        batch, frame, frequency = feature_mix.shape
        batch, num_spk, nsample = sig_ref.shape
        feature_mix = feature_mix.reshape(frame, frequency)
        embedding = embedding.reshape(frame, frequency, -1)
        m = np.max(feature_mix) - 40/20
        emb = embedding[feature_mix>=m,:]
        label = KMeans(n_clusters=num_spk, random_state=0).fit_predict(emb)
        mask = np.zeros((num_spk, frame, frequency))
        mask[0, feature_mix>=m] = label
        mask[1, feature_mix>=m] = 1-label
        stft_est = stft_mix * mask
        sig_est = np.zeros((batch, num_spk, nsample))
        for i in range(num_spk):
            sig_est[0, i] = librosa.core.istft(stft_est[i].T, hop_length=64, length=nsample)
        sig_est = torch.tensor(sig_est).to(self.device)
        return sig_est

    def eval(self):

        sdrs = AverageMeter()
        self.model.eval()
        for i, data in enumerate(self.data_loader):
            input, label = data
            output = self.model(input)
            feature_mix, = input
            embedding, = output
            stft_r_mix, stft_i_mix, sig_ref = label
            sig_est = self.get_est_sig([feature_mix, embedding, stft_r_mix, stft_i_mix, sig_ref])
            sdr = evaluate.batch_SDR_torch(sig_est, sig_ref)
            sdrs.update(sdr)
            print("SDR: %.2f"%(sdrs.avg), end='\r')




def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path",
                        help='The path to the config file. e.g. python train.py --config configs/dc_config.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = tester(args)
    t.eval()


if __name__ == "__main__":
    main()
