from attrdict import AttrDict
from .basic import AverageMeter
import argparse, json, os, time, torch


class trainer:
    def __init__(self, args):
        """
        args: a dictionary containing
            model_name(str): the name of the model
            model(nn.Module): the module object
            data_loader(DataLoader): the PyTorch built-in DataLoader object
            loss_fn(function): the loss function
            resume_from_checkpoint(bool): check if resume the training from a checkpoint, False by default
            checkpoint_path(str): the path to the saved dictionary
            device(torch.device): the device for training, cpu by default
        """

        if "resume_from_checkpoint" in args and args.resume_from_checkpoint=="True":
            self.resume_from_checkpoint = True
        else:
             self.resume_from_checkpoint = False

        self.device = args.device
        self.model_name = args.model_name
        self.train_loader = args.train_loader
        self.valid_loader = args.valid_loader
        self.loss_fn = args.loss_fn

        # build model
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint(args.checkpoint_path)
        else:
            self.model = args.model
            self.optimizer = args.optimizer
            self.epoch = 0
            self.min_loss = float("inf")
            self.early_stop_count = 0
        print("Loaded the model...")

        self.num_epoch = args.num_epoch
        self.checkpoint_path = args.checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def resume_from_checkpoint(self, checkpoint_path):
        saved_dict = torch.load(checkpoint_path+'/final.mdl')
        self.model = saved_dict["model"]
        self.model = self.model.to(self.device)
        self.epoch = saved_dict["epoch"]
        self.min_loss = saved_dict["cv_loss"]
        self.early_stop_count = saved_dict["early_stop_count"]

    def run(self):
        for epoch in range(self.epoch, self.num_epoch):
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
        init_time = time.time()
        end = init_time
        for i, data in enumerate(self.train_loader):
            input, label = data
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            self.optimizer.zero_grad()
            loss_avg.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            times.update(time.time()-end)
            end = time.time()
            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f/%.2f seconds'%(epoch, i+1,len_d,losses.avg, end-init_time, times.avg*len_d), end='\r')
        print("\n")

    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        len_d = len(self.valid_loader)
        init_time = time.time()
        end = init_time
        for i, data in enumerate(self.valid_loader):
            begin = time.time()
            input, label = data
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss_avg = torch.mean(loss)
            losses.update(loss_avg.item())
            times.update(time.time()-end)
            end = time.time()
            print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f/%.2f seconds'%(epoch, i+1,len_d,losses.avg, end-init_time, times.avg*len_d), end='\r')
        print("\n")
        if losses.avg < self.min_loss:
            self.early_stop_count = 0
            self.min_loss = losses.avg
            saved_dict = {
                'model': self.model.state_dict(),
                'epoch': epoch,
                'optimizer': self.optimizer,
                'cv_loss': self.min_loss,
                "early_stop_count": self.early_stop_count
            }
            torch.save(saved_dict,self.checkpoint_path+"/final.mdl")
            print("Saved new model")
        else:
            self.early_stop_count += 1


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
