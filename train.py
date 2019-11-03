from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, nn, numpy as np, os, time, torch


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


class trainer:
    def __init__(self, args):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        self.dataset = args.dataset
        if args.cuda_option == "True":
            print("GPU mode on...")
            available_device = get_free_gpu()
            print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')
        # build model
        self.model = self.init_model(args.model_name, args.model_options)
        print("Loaded the model...")
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        print("Built the loss function...")
        # build optimizer
        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        print("Built the optimizer...")
        # build DataLoaders
        if args.dataset == "wsj0-2mix":
            self.train_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
            self.valid_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)
        elif args.dataset == "daps":
            self.train_loader = data.daps_enhance_dataloader(args.train_num_batch, args.feature_options, 'train', args.cuda_option, self.device)
            self.valid_loader = data.daps_enhance_dataloader(args.vaildate_num_batch, args.feature_options, 'validation', args.cuda_option, self.device)
        elif args.dataset == "edinburgh_tts":
            self.train_loader = data.edinburgh_tts_dataloader(args.model_name, args.feature_options, 'train', args.cuda_option, self.device)
            self.valid_loader = data.edinburgh_tts_dataloader(args.model_name, args.feature_options, 'validation', args.cuda_option, self.device)

        # training options
        self.num_epoch = args.num_epoch
        self.output_path = args.output_path+'/%s_%s_%s'%(self.model_name, self.dataset, self.loss_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.min_loss = float("inf")
        self.early_stop_count = 0

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


    def build_lossfn(self, fn_name):
        return get_lossfns()[fn_name]


    def build_optimizer(self, params, optimizer_options):
        if optimizer_options.name == "adam":
            return torch.optim.Adam(params, lr=optimizer_options.lr)
        if optimizer_options.name == "sgd":
            return torch.optim.SGD(params, lr=optimizer_options.lr, momentum=0.9)
        if optimizer_options.name == "rmsprop":
            return torch.optim.RMSprop(params, lr=optimizer_options.lr)

    def run(self):
        for epoch in range(self.num_epoch):
            self.train(epoch)
            self.validate(epoch)
            if self.early_stop_count == 5:
                print("Model stops improving, stop the training")
                break
        print("Model training is finished.")


    def train(self, epoch):
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        self.model.train()
        len_d = len(self.train_loader)
        end = time.time()
        for i, data in enumerate(self.train_loader):
            input, label = data
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            self.optimizer.zero_grad()
            loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            times.update(time.time()-end)
            end = time.time()
            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len_d,losses.avg, times.avg*len_d), end='\r')
        print("\n")


    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        len_d = len(self.valid_loader)
        end = time.time()
        for i, data in enumerate(self.valid_loader):
            begin = time.time()
            input, label = data
            if torch.sum(label[0]) < 1:
                continue
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            times.update(time.time()-end)
            end = time.time()
            print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len_d,losses.avg, times.avg*len_d), end='\r')
        print("\n")
        if losses.avg < self.min_loss:
            self.early_stop_count = 0
            self.min_loss = losses.avg
            torch.save(self.model,self.output_path+"/model.epoch%d"%epoch)
            print("Saved new model")
        else:
            self.early_stop_count += 1


    def evaluate(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path",
                        help='The path to the config file. e.g. python train.py --config configs/dc_config.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = trainer(args)
    t.run()


if __name__ == "__main__":
    main()
