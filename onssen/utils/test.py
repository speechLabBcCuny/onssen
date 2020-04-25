from attrdict import AttrDict
from .basic import AverageMeter
import argparse, json, os, time, torch
from ..evaluate import batch_SDR_torch


class tester:
    def __init__(self, args):
        """
        args: a dictionary containing
            model_name(str): the name of the model
            model(nn.Module): the module object
            test_loader(DataLoader): the PyTorch built-in DataLoader object
        """
        self.model_name = args.model_name
        self.test_loader = args.test_loader
        self.device = args.device
        
        # build model
        saved_dict = torch.load(args.checkpoint_path+'/final.mdl')
        self.model = saved_dict["model"]
        self.model = self.model.to(self.device)
        print("Loaded the model...")

    def get_est_sig(self, input, label, output):
        pass

    def eval(self):
        sdrs = AverageMeter()
        self.model.eval()
        for i, data in enumerate(self.test_loader):
            input, label = data
            output = self.model(input)
            sig_est, sig_ref = self.get_est_sig(input, label, output)
            sdr = batch_SDR_torch(sig_est, sig_ref)
            sdrs.update(sdr)
            print("SDR: %.2f"%(sdrs.avg), end='\r')

        print("\n")


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path",
                        help='The path to the config file. e.g. python train.py --config config.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = tester(args)
    t.eval()


if __name__ == "__main__":
    main()
