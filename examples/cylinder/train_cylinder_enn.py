"""
=====
Training embedding model for the flow around cylinder numerical example.
This is a built-in model from the paper.

Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import sys
import logging
import torch
import time
from torch.optim.lr_scheduler import ExponentialLR

from trphysx.config.configuration_auto import AutoPhysConfig
from trphysx.embedding.embedding_auto import AutoEmbeddingModel
from trphysx.viz.viz_auto import AutoViz
from trphysx.embedding.training import *

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    start_time = time.time()

    sys.argv = sys.argv + ["--exp_name", "cylinder"]
    # sys.argv = sys.argv + ["--training_h5_file", "./data/cylinder_training.hdf5"]
    sys.argv = sys.argv + ["--training_h5_file", "/scratch1/jordan/amr_physx/examples/cylinder/data/cylinder_train.hdf5"]
    # sys.argv = sys.argv + ["--eval_h5_file", "/data/cylinder_valid.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file", "/scratch1/jordan/amr_physx/examples/cylinder/data/cylinder_valid.hdf5"]
    sys.argv = sys.argv + ["--batch_size", "6"]
    sys.argv = sys.argv + ["--block_size", "4"]
    sys.argv = sys.argv + ["--n_train", "27"]
    sys.argv = sys.argv + ["--n_eval", "6"]
    sys.argv = sys.argv + ["--epochs", "50"]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    args = EmbeddingParser().parse()
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("args.device ", args.device)
    logger.info("Torch device: {}".format(args.device))
    print(torch.cuda.get_arch_list())
    print(torch.version.cuda)

    # Load transformer config file
    config = AutoPhysConfig.load_config(args.exp_name)
    data_handler = AutoDataHandler.load_data_handler(args.exp_name)
    viz = AutoViz.load_viz(args.exp_name, plot_dir=args.plot_dir)

     # Set up data-loaders
    training_loader = data_handler.createTrainingLoader(
                        args.training_h5_file, 
                        block_size=args.block_size, 
                        stride=args.stride, 
                        ndata=args.n_train, 
                        batch_size=args.batch_size)
    testing_loader = data_handler.createTestingLoader(
                        args.eval_h5_file, 
                        block_size=32, 
                        ndata=args.n_eval, 
                        batch_size=8)

    # print("stop here 2")
    # exit()

    # Set up model
    model = AutoEmbeddingModel.init_trainer(args.exp_name, config).to(args.device)

    mu, std = data_handler.norm_params
    model.embedding_model.mu = mu.to(args.device)
    model.embedding_model.std = std.to(args.device)
    if args.epoch_start > 1:
        model.load_model(args.ckpt_dir, args.epoch_start)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*0.995**(args.epoch_start-1), weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler), viz)
    trainer.train(training_loader, testing_loader)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds.")

