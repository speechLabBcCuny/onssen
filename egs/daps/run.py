from onssen import data, loss, nn, utils
from attrdict import AttrDict
import torch
import json


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path",
                        help='The path to the config file. e.g. python run.py --config dc_config.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    device = torch.device(args.device)
    args.model = onssen.nn.enhance(args.model_options)
    args.model.to(device)
    args.train_loader = data.daps_enhance_dataloader(args.train_num_batch, args.feature_options, 'train', args.cuda_option, self.device)
    args.valid_loader = data.daps_enhance_dataloader(args.vaildate_num_batch, args.feature_options, 'validation', args.cuda_option, self.device)
    args.optimizer = utils.build_optimizer(args.model.parameters(), args.optimizer_options)
    args.loss_fn = loss.loss_mask_msa
    trainer = onssen.utils.trainer(args)
    trainer.run()

    tester = onssen.utils.tester(args)
    tester.eval()


if __name__ == "__main__":
    main()
