import sys
sys.path.append('/home/near/onssen/')
from onssen import utils


class evaluate(utils.tester):
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
        feature_mix, = input
        embedding, = output
        stft_r_mix, stft_i_mix, sig_ref = label

        stft_r_mix = stft_r_mix.detach().cpu().numpy()
        stft_i_mix = stft_i_mix.detach().cpu().numpy()
        
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
        return sig_est, sig_ref