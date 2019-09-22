import argparse
from attrdict import AttrDict
from data.features import wsj0_2mix_dataloader
import json
from losses.loss_util import get_lossfns
import nn
import torch
import time
import os
from utils import AverageMeter


class trainer:
    def __init__(self, args):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        # build model
        self.model = self.init_model(args.model_name, args.model_options, args.cuda)
        print("Loaded the model...")
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        print("Built the loss function...")
        # build optimizer
        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        print("Built the optimizer...")
        # build DataLoaders
        if args.num_speaker == 2:
            self.train_loader = wsj0_2mix_dataloader(args.model_name, args.feature_options, 'tt', args.cuda)
            self.valid_loader = wsj0_2mix_dataloader(args.model_name, args.feature_options, 'cv', args.cuda)
        # training options
        self.num_epoch = args.num_epoch
        self.output_path = args.output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.min_loss = float("inf")
        self.early_stop_count = 0

    def init_model(self, model_name, model_options, cuda):
        assert model_name is not None, "Model name must be defined!"
        assert model_name in ["dc", "chimera", "chimera++", "phase"],\
        "Model name is not supported! Must be one of (dc, chimera, chimera++, phase)"
        if model_name == "dc":
            model = nn.deep_clustering(
                model_options.input_dim,
                model_options.get("hidden_dim",None),
                model_options.get("embedding_dim",None),
                model_options.get("num_layers",None),
                model_options.get("dropout",0.3),
            )
        if model_name == "chimera" or model_name == "chimera++":
            model = nn.chimera(
                model_options.input_dim,
                model_options.get("hidden_dim",None),
                model_options.get("embedding_dim",None),
                model_options.get("num_layers",None),
                model_options.get("dropout",0.3),
            )
        if model_name == "phase":
            model = nn.phase_net(
                model_options.input_dim,
                model_options.get("hidden_dim",None),
                model_options.get("embedding_dim",None),
                model_options.get("num_layers",None),
                model_options.get("dropout",0.3),
            )
            
        if cuda == "True":
            print("GPU mode on...")
            model.cuda()
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
        for i, data in enumerate(self.train_loader):
            begin = time.time()
            input, label = data
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            self.optimizer.zero_grad()
            loss_avg.backward()
            self.optimizer.step()
            end = time.time()
            times.update(end-begin)
            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len_d,losses.avg, times.avg*len_d), end='\r')
        print("\n")


    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        len_d = len(self.valid_loader)
        for i, data in enumerate(self.valid_loader):
            begin = time.time()
            input, label = data
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            end = time.time()
            times.update(end-begin)
            print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len_d,losses.avg, times.avg*len_d), end='\r')
        print("\n")
        if losses.avg < self.min_loss:
            self.early_stop_count = 0
            self.min_loss = losses.avg
            torch.save(self.model,self.output_path+"/model_%s_%s.epoch%d"%(self.model_name, self.loss_name, epoch))
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
