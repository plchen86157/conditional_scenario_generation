import os
import sys
from os.path import join as pjoin
from datetime import datetime

import argparse

from core.dataloader.argoverse_loader import Argoverse, GraphData, ArgoverseInMem
from core.dataloader.argoverse_loader_v2 import ArgoverseInMem as ArgoverseInMemv2
from core.trainer.tnt_trainer import TNTTrainer
from torch.utils.tensorboard import SummaryWriter

sys.path.append("core/dataloader")


def train(args):
    """
    script to train the tnt
    :param args:
    :return:
    """
    #time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    time_stamp = str(datetime.now())[:19]
    #print("time_stamp:", time_stamp)
    #print(str(datetime.now())[:19])
    print(pjoin(args.data_root, "train_intermediate"))
    train_set = ArgoverseInMemv2(pjoin(args.data_root, "train_intermediate")).shuffle()
    eval_set = ArgoverseInMemv2(pjoin(args.data_root, "val_intermediate"))
    test_set = ArgoverseInMemv2(pjoin(args.data_root, "test_intermediate"))

    # init output dir
    
    output_dir = pjoin(args.output_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        #raise Exception("The output folder does exists and is not empty! Check the folder.")
        pass
    else:
        os.makedirs(output_dir)
    # init trainer
    trainer = TNTTrainer(
        trainset=train_set,
        evalset=eval_set,
        testset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_update_freq=args.lr_update_freq,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        num_global_graph_layer=args.num_glayer,
        aux_loss=args.aux_loss,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        log_freq=args.log_freq,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None, 
        num_subgraph_layers=args.num_subgraph_layers,
        subgraph_width=args.subgraph_width,
        global_graph_width=args.global_graph_width,
        target_pred_hid=args.target_pred_hid,
        positive_weight=args.positive_weight,
        nearest_subgraph=args.nearest_subgraph
    )
    # resume minimum eval loss
    #min_eval_loss = trainer.min_eval_loss
    max_score = 0
    min_eval_loss = 2147483647
    min_train_loss = 2147483647

    # training
    for iter_epoch in range(args.n_epoch):
        train_loss = trainer.train(iter_epoch)

        eval_loss = trainer.eval(iter_epoch)
        #print(train_loss, eval_loss)
        #exit()
        if eval_loss < min_eval_loss:
            # save the model when a lower eval_loss is found
            min_eval_loss = eval_loss

            #trainer.save(iter_epoch, min_eval_loss)
            trainer.save_model("min_val_loss")
        if train_loss < min_train_loss:
            # save the model when a lower eval_loss is found
            min_train_loss = train_loss

            #trainer.save(iter_epoch, min_train_loss)
            trainer.save_model("min_train_loss")
        # if (iter_epoch + 1) % 50 == 0:
        #     trainer.save_model(str(iter_epoch + 1))
        ########################## Overfit #################################
        #iter_epoch = -1
        if (iter_epoch + 1) > 10000:
            if (iter_epoch + 1) % args.n_epoch == 0:
                _, _, _ = trainer.eval_when_training(split='train', iter_epoch=iter_epoch, output_dir=output_dir)
        #trainer.save(iter_epoch, min_eval_loss)
        #trainer.save_model("best_eval")
        ###################################################################### 
        if (iter_epoch + 1) > 0:
            if (iter_epoch + 1) % args.n_epoch == 0:
                IOU_over_50, TP_IOU_over_50, tp_selection = trainer.eval_when_training(split='val', iter_epoch=iter_epoch, output_dir=output_dir)
                tp_selection_lambda = 1 #0.25
                final_score = IOU_over_50 + TP_IOU_over_50 + tp_selection * tp_selection_lambda
                print("EVAL:", IOU_over_50, TP_IOU_over_50, tp_selection)
                if final_score > max_score:
                    max_score = final_score
                    #trainer.save(iter_epoch, eval_loss)
                    trainer.save_model("best_eval")
        ######################################################################

    #trainer.test()
    trainer.save_model("final")
    

    ### test ###
    #test_set = ArgoverseInMemv2(pjoin(args.data_root, "{}_intermediate".format('test')))
    trainer = TNTTrainer(
        trainset=train_set,
        evalset=train_set,
        testset=train_set,
        batch_size=256,
        num_workers=16,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        ckpt_path=None,
        model_path=output_dir + '/min_train_loss_TNT.pth',
        num_global_graph_layer=args.num_glayer,
        num_subgraph_layers=args.num_subgraph_layers,
        subgraph_width=args.subgraph_width,
        global_graph_width=args.global_graph_width,
        target_pred_hid=args.target_pred_hid,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_update_freq=args.lr_update_freq,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.adam_weight_decay,
        positive_weight=args.positive_weight,
        nearest_subgraph=args.nearest_subgraph
    )

    #_ = trainer.test(m=1, split='train', miss_threshold=2.0, save_pred=True, convert_coordinate=True, save_folder=output_dir)
    
    # # _, _, _ = trainer.eval_when_training(split='train', iter_epoch=-1, output_dir=output_dir)

    trainer = TNTTrainer(
        trainset=train_set,
        evalset=eval_set,
        testset=eval_set,
        batch_size=256,
        num_workers=16,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        ckpt_path=None,
        model_path=output_dir + '/best_eval_TNT.pth',
        num_global_graph_layer=args.num_glayer,
        num_subgraph_layers=args.num_subgraph_layers,
        subgraph_width=args.subgraph_width,
        global_graph_width=args.global_graph_width,
        target_pred_hid=args.target_pred_hid,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_update_freq=args.lr_update_freq,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.adam_weight_decay,
        positive_weight=args.positive_weight,
        nearest_subgraph=args.nearest_subgraph
    )
    _ = trainer.test(m=1, split='val', miss_threshold=2.0, save_pred=True, convert_coordinate=True, save_folder=output_dir)

    trainer = TNTTrainer(
        trainset=train_set,
        evalset=test_set,
        testset=test_set,
        batch_size=256,
        num_workers=16,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        ckpt_path=None,
        model_path=output_dir + '/best_eval_TNT.pth',
        num_global_graph_layer=args.num_glayer,
        num_subgraph_layers=args.num_subgraph_layers,
        subgraph_width=args.subgraph_width,
        global_graph_width=args.global_graph_width,
        target_pred_hid=args.target_pred_hid,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_update_freq=args.lr_update_freq,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.adam_weight_decay,
        positive_weight=args.positive_weight,
        nearest_subgraph=args.nearest_subgraph
    )
    _ = trainer.test(m=1, split='test', miss_threshold=2.0, save_pred=True, convert_coordinate=True, save_folder=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", required=False, type=str, default="nuscenes_data/interm_data",
                        help="root dir for datasets")
    parser.add_argument("-o", "--output_dir", required=False, type=str, default="run/tnt/",
                        help="ex)dir to save checkpoint and model")
    parser.add_argument("-a", "--aux_loss", action="store_true", default=True,
                        help="Training with the auxiliary recovery loss")

    parser.add_argument("-b", "--batch_size", type=int, default=2,
                        help="number of batch_size")
    parser.add_argument("-e", "--n_epoch", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=16,
                        help="dataloader worker size")

    parser.add_argument("-c", "--with_cuda", action="store_true", default=False,
                        help="training with CUDA: true, or false")
    # parser.add_argument("-cd", "--cuda_device", type=int, default=[1, 0], nargs='+',
    #                     help="CUDA device ids")
    parser.add_argument("-cd", "--cuda_device", type=int, nargs='+', default=[],
                        help="CUDA device ids")
    parser.add_argument("--log_freq", type=int, default=2,
                        help="printing loss every n iter: setting n")
    # parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate of adam")
    parser.add_argument("-we", "--warmup_epoch", type=int, default=20,
                        help="the number of warmup epoch with initial learning rate, after the learning rate decays")
    parser.add_argument("-luf", "--lr_update_freq", type=int, default=5,
                        help="learning rate decay frequency for lr scheduler")
    parser.add_argument("-ldr", "--lr_decay_rate", type=float, default=0.9, help="lr scheduler decay rate")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        help="resume a model state for fine-tune")
    parser.add_argument("-l", "--num_glayer", type=int, default=1,
                        help="number of global graph layers")
    parser.add_argument("--num_subgraph_layers", type=int, default=3)
    parser.add_argument("--subgraph_width", type=int, default=64)
    parser.add_argument("--global_graph_width", type=int, default=64)
    parser.add_argument("--target_pred_hid", type=int, default=64)
    
    parser.add_argument("-w_p", "--positive_weight", type=float, default=10)
    parser.add_argument("-n_s", "--nearest_subgraph", type=int, default=10)

    args = parser.parse_args()
    train(args)
