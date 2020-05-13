import sys
sys.path.append('../../../../../onssen/')
sys.path.append('../')
from onssen import data, loss, nn, utils
from attrdict import AttrDict
import torch
import json
from evaluate import tester_chimera

def main():
    config_path = './config.json'
    with open(config_path) as f:
        args = json.load(f)
        args = AttrDict(args)
    device = torch.device(args.device)
    args.model = nn.chimera(**(args['model_options']))
    args.model.to(device)
    args.train_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'tr', device)
    args.valid_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'cv', device)
    args.test_loader = data.wsj0_2mix_dataloader(args.model_name, args.feature_options, 'tt', device)
    args.optimizer = utils.build_optimizer(args.model.parameters(), args.optimizer_options)
    args.loss_fn = loss.loss_chimera_msa
    trainer = utils.trainer(args)
    trainer.run()
    tester = tester_chimera(args)
    tester.eval()


if __name__ == "__main__":
    main()
