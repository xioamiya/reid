import data
import loss
import torch
import model
from trainer import Trainer
from option import args
import utils.utility as utility
import os

if __name__ == "__main__":
    ckpt = utility.checkpoint(args)

    loader = data.Data(args)
    model = model.Model(args, ckpt)
    loss = loss.Loss(args, ckpt) if not args.test_only else None

    trainer = Trainer(args, model, loss, loader, ckpt)

    n = 0
    while not trainer.terminate():
        n += 1
        trainer.train()
        # print("n=")
        # print(n)
        if args.test_every != 0 and n % args.test_every == 0:
            trainer.test()
            # print("hello")
