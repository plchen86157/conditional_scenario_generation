import os
import sys
from os.path import join as pjoin
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np

# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from core.dataloader.dataset import GraphDataset
# from core.dataloader.argoverse_loader import Argoverse, GraphData, ArgoverseInMem
from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem
from core.trainer.tnt_trainer import TNTTrainer
from torch.utils.tensorboard import SummaryWriter

sys.path.append("core/dataloader")

def visualize_loss(args):
    data = pd.read_csv(args.loss_curve_path + 'training_loss.csv')
    print(data['epoch'].values[-1])
    epoch_range = np.arange(0, data['epoch'].values[-1] + 1, 1)
    train_curve = np.array(data[['train_loss']])
    cls_curve = np.array(data[['cls_loss']])
    offset_curve = np.array(data[['offset_loss']])
    yaw_curve = np.array(data[['yaw_loss']])
    ttc_curve = np.array(data[['ttc_loss']])
    plt.plot(epoch_range, train_curve, label = "train_loss")
    plt.plot(epoch_range, cls_curve, label = "cls_loss")
    plt.plot(epoch_range, offset_curve, label = "offset_loss")
    plt.plot(epoch_range, yaw_curve, label = "yaw_loss")
    plt.plot(epoch_range, ttc_curve, label = "ttc_loss")
    plt.legend()
    plt.show()


def test(args):
    """
    script to test the tnt model
    "param args:
    :return:
    """
    # config
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = pjoin(args.save_dir, time_stamp)
    # if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    #     raise Exception("The output folder does exists and is not empty! Check the folder.")
    # else:
    #     os.makedirs(output_dir)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # data loading
    train_set = ArgoverseInMem(pjoin(args.data_root, "train_intermediate")).shuffle()
    eval_set = ArgoverseInMem(pjoin(args.data_root, "val_intermediate"))
    try:
        test_set = ArgoverseInMem(pjoin(args.data_root, "{}_intermediate".format(args.split)))
    except:
        raise Exception("Failed to load the data, please check the dataset!")
    #init trainer
    trainer = TNTTrainer(
        trainset=test_set,
        evalset=test_set,
        testset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=args.save_dir,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None,
        num_global_graph_layer=args.num_global_graph_layer
    )
    
    # trainer.other_visualization()
    IOU_over_50 = trainer.test(args.top_m_bbox, split=args.split, miss_threshold=2.0, save_pred=True, convert_coordinate=True, save_folder=args.save_dir)
    
    # trainer = TNTTrainer(
    #     trainset=train_set,
    #     evalset=eval_set,
    #     testset=test_set,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     lr=0.001,
    #     warmup_epoch=500,
    #     lr_update_freq=100,
    #     lr_decay_rate=0.5,
    #     weight_decay=0.01,
    #     betas=(0.9, 0.999),
    #     num_global_graph_layer=1,
    #     aux_loss=True,
    #     with_cuda=args.with_cuda,
    #     cuda_device=args.cuda_device,
    #     save_folder=output_dir,
    #     log_freq=2,
    #     ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
    #     model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None, 
    #     num_subgraph_layers=3,
    #     subgraph_width=64,
    #     global_graph_width=64,
    #     target_pred_hid=64,
    # )
    # IOU_over_50, average_sim, tp_selection = trainer.eval_when_training(args.top_m_bbox, split=args.split, iter_epoch=0, output_dir=output_dir, miss_threshold=2.0, save_pred=True, convert_coordinate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", type=str, default="nuscenes_data/interm_data",
                        help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="test")

    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=16,
                        help="dataloader worker size")
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=[0], nargs='+',
                        help="CUDA device ids")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        default="run/tnt/2024-11-13 04:15:15/final_TNT.pth")
    parser.add_argument("-sd", "--save_dir", type=str, default="run/tnt/2024-11-13 04:15:15")
    parser.add_argument("--loss_curve_path", type=str,
                        default="run/tnt/2024-11-13 04:15:15/final_TNT.pth")
    parser.add_argument("--top_m_bbox", type=int,
                        default=5)
    parser.add_argument("--num_global_graph_layer", type=int,
                        default=1)
    args = parser.parse_args()
    #visualize_loss(args)
    test(args)
