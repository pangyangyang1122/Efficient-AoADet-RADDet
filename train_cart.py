import torch.optim
import util.loader as loader
import argparse
import os
import sys
import numpy as np
from engine.launch import launch
from model.model import RADDet, FPNDet
from utils.collect_env import collect_env_info
from utils.dist_utils import get_rank
from dataset.radar_dataset_3d import RararDataset3D
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer_utils import LinearWarmupCosineAnnealingLR
from metrics import mAP
import utils.dist_utils as dist_utils
from model.model_cart import RADDetCart
from model.yolo_loss import yoloheadToPredictions2D, nms2DOverClass
import pkbar
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    # initialization
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(env_str)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, "configFPN.json")
    config = loader.readConfig(config_file_name=config_path)
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_loss = config["LOSS"]
    resume = args.resume

    # load anchor boxes with order
    alpha_init = config_loss["FFT_loss_init"]
    alpha_final = config_loss["FFT_loss_final"]
    anchors_path = os.path.join(dir_path, "anchors.txt")
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file=anchors_path)
    num_classes = len(config_data["all_classes"])

    anchors_cart_path = os.path.join(dir_path, "anchors_cartboxes.txt")
    anchor_boxes_cart = loader.readAnchorBoxes(anchor_boxes_file=anchors_cart_path)

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = FPNDet(config_model, config_data, config_train, anchor_boxes)
    # model.to(device)
    ####直接训练2D检测不用3D的预训练了###
    # print(f"Load pretrained model from {args.backbone_resume_from}")
    # model.load_state_dict(torch.load(args.backbone_resume_from), strict=False)

    model_bk = model.backbone
    model_RADec = model.RA_decoder

    model_cart = RADDetCart(
        config_model,
        config_data,
        config_train,
        anchor_boxes_cart,
        config_model["bk_output_size"],
        device,
        backbone=model_bk,
        RA_decoder=model_RADec,
    )
    t_params = sum(p.numel() for p in model_cart.parameters())
    print("Network Parameters: ",t_params)
    model_cart.to(device)

    train_dataset = RararDataset3D(
        config_data=config_data,
        config_train=config_train,
        config_model=config_model,
        headoutput_shape=config_data["headoutput_shape"],
        anchors=anchor_boxes,
        anchors_cart=anchor_boxes_cart,
        cart_shape=config_data["cart_shape"],
        dType="train",
        Input_type=config_data["input_type"],
    )

    validate_dataset = RararDataset3D(
        config_data=config_data,
        config_train=config_train,
        config_model=config_model,
        headoutput_shape=config_data["headoutput_shape"],
        anchors=anchor_boxes,
        anchors_cart=anchor_boxes_cart,
        cart_shape=config_data["cart_shape"],
        dType="validate",
        Input_type=config_data["input_type"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_train["batch_size"] // args.num_gpus,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=config_train["batch_size"] // args.num_gpus,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dataset = RararDataset3D(
        config_data=config_data,
        config_train=config_train,
        config_model=config_model,
        headoutput_shape=config_data["headoutput_shape"],
        anchors=anchor_boxes,
        anchors_cart=anchor_boxes_cart,
        cart_shape=config_data["cart_shape"],
        dType="test",
        Input_type=config_data["input_type"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config_train["batch_size"] // args.num_gpus,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    if get_rank() == 0:
        ### NOTE: training settings ###
        # logdir = os.path.join(config_train["log_dir"], "cartesian",
        #                       "b_" + str(config_train["batch_size"]) + "lr_" + str(config_train["learningrate_init"]))
        logdir = os.path.join(
            config_train["log_dir"],
            config["NAME"]
            +"_"+config_data["input_type"]
            + "-b_"
            + str(config_train["batch_size"])
            # + "-lr_"
            # + str(config_train["learningrate_init"])
            # + "_cartesian",
        )
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(log_dir=logdir)
        log_specific_dir = os.path.join(logdir, "ckpt")
        if not os.path.exists(log_specific_dir):
            os.makedirs(log_specific_dir)

    optimizer = torch.optim.Adam(
        params=model_cart.parameters(),
        lr=config_train["learningrate_init"],
        betas=(0.9, 0.99),
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config_train["warmup_steps"],
        eta_min=config_train["learningrate_end"],
        max_epochs=config_train["epochs"],
        warmup_start_lr=config_train["learningrate_init"],
    )
    step, best_val = 0, 0.0
    startEpoch = 0

    log_name = os.path.join(logdir, "log.txt")
    if not os.path.exists(log_name) or os.path.getsize(log_name) == 0:
    # 如果文件不存在或为空，首先写入标题行
        with open(log_name, "w") as f:
            f.write("Epoch  lr  Ttotal_loss  TFFT_loss   TDet_loss  Vtotal_loss  VFFT_loss  Vdet_loss  MAP  ap_person  ap_bicycle  ap_car  ap_motorcycle  ap_bus  ap_truck\n")

    if resume:
        print("===========  Resume training  ==================:")
        dict = torch.load(resume)
        model_cart.load_state_dict(dict["net_state_dict"])
        optimizer.load_state_dict(dict["optimizer"])
        scheduler.load_state_dict(dict["scheduler"])
        startEpoch = dict["epoch"] + 1
        step = dict["global_step"]
        best_mAP = 0.4545
        print("       ... Start at epoch:", startEpoch)

    for epoch in range(startEpoch, config_train["epochs"]):
        kbar = pkbar.Kbar(
            target=len(train_loader),
            epoch=epoch,
            num_epochs=config_train["epochs"],
            width=20,
            always_stateful=False,
        )  # Show progress bar

        ###################
        ## Training loop ##
        ###################
        model_cart.train()
        Ttotal_loss = 0.0
        TFFT_loss = 0.0
        TDet_loss = 0.0
        Tbox_loss = 0.0
        Tconf_loss = 0.0
        Tcategory_loss = 0.0
        if alpha_init - epoch >= alpha_final:
            alpha = alpha_init - epoch
        else: alpha = alpha_final

        for d in tqdm(train_loader):
            data, Label_RA_map, label, raw_boxes = d
            data = data.to(device)    #6，8，256，64
            Label_RA_map = Label_RA_map.to(device)
            label = label.to(device)
            raw_boxes = raw_boxes.to(device)
            # print(data.shape, label.shape, raw_boxes.shape, data.device)
            pred_raw, ARD_pred = model_cart(data)
            pred = model_cart.decodeYolo(pred_raw)
            DRA_pred = ARD_pred.permute(0,3,2,1)
            # DRA_pred_cat = torch.cat((DRA_pred.real,DRA_pred.imag),dim=1)
            # DRA_label_cat = torch.cat((Label_RA_map.real,Label_RA_map.imag),dim=1)
            pred_RA_map = torch.sum((DRA_pred.real** 2 + DRA_pred.imag ** 2), dim=1)
            FFT_loss = F.l1_loss(pred_RA_map, Label_RA_map)
            total_loss_D, box_loss, conf_loss, category_loss = model_cart.loss(
                pred_raw, pred, label, raw_boxes[..., :4]
            )
            total_loss = total_loss_D + alpha * FFT_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_r, FFT_loss_r, Det_loss_r, box_loss_r, conf_loss_r, category_loss_r = (
                total_loss.cpu().detach(),
                FFT_loss.cpu().detach(),
                total_loss_D.cpu().detach(),
                box_loss.cpu().detach(),
                conf_loss.cpu().detach(),
                category_loss.cpu().detach(),
            )

            if get_rank() == 0:
                writer.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], global_step=step
                )
                writer.add_scalar("loss/total_loss", total_loss_r, global_step=step)
                writer.add_scalar("loss/FFT_loss", FFT_loss_r, global_step=step)
                writer.add_scalar("loss/Det_loss", Det_loss_r, global_step=step)
                writer.add_scalar("loss/box_loss", box_loss_r, global_step=step)
                writer.add_scalar("loss/conf_loss", conf_loss_r, global_step=step)
                writer.add_scalar("loss/category_loss", category_loss_r, global_step=step)
                writer.flush()
            step += 1
            Ttotal_loss += total_loss_r
            TFFT_loss += FFT_loss_r
            TDet_loss += Det_loss_r
            Tbox_loss += box_loss_r
            Tconf_loss += conf_loss_r
            Tcategory_loss += category_loss_r
        if get_rank() == 0:
            print(
                "=======> epochs: %4d, train step: %4d, lr: %.10f, total_loss: %4.2f, FFT_loss: %4.2f, Det_loss: %4.2f, "
                "box_loss: %4.2f, conf_loss: %4.2f, category_loss: %4.2f"
                % (
                    epoch,
                    step,
                    optimizer.param_groups[0]["lr"],
                    Ttotal_loss / len(train_loader.dataset),
                    TFFT_loss / len(train_loader.dataset),
                    TDet_loss / len(train_loader.dataset),
                    Tbox_loss / len(train_loader.dataset),
                    Tconf_loss / len(train_loader.dataset),
                    Tcategory_loss / len(train_loader.dataset),
                )
            )

        ####开始验证
        if get_rank() == 0:
            print(f"epochs: {epoch}, start validation")
            mean_ap_test = 0.0
            ap_all_class_test = []
            ap_all_class = []
            total_losstest = []
            FFT_losstest = []
            Det_losstest = []
            box_losstest = []
            conf_losstest = []
            category_losstest = []

            for class_id in range(num_classes):
                ap_all_class.append([])
            model_cart.eval()
            for d in validate_loader:
                with torch.no_grad():
                    data, Label_RA_map, label, raw_boxes = d
                    data = data.to(device)
                    Label_RA_map = Label_RA_map.to(device)
                    label = label.to(device)
                    raw_boxes = raw_boxes.to(device)
                    # print(data.shape, label.shape, raw_boxes.shape, data.device)
                    pred_raw, ARD_pred = model_cart(data)
                    pred = model_cart.decodeYolo(pred_raw)
                    DRA_pred = ARD_pred.permute(0,3,2,1)
                    # DRA_pred_cat = torch.cat((DRA_pred.real,DRA_pred.imag),dim=1)
                    # DRA_label_cat = torch.cat((Label_RA_map.real,Label_RA_map.imag),dim=1)
                    pred_RA_map = torch.sum((DRA_pred.real** 2 + DRA_pred.imag ** 2), dim=1)
                    FFT_loss = F.l1_loss(pred_RA_map, Label_RA_map)
                    Det_loss, box_loss, conf_loss, category_loss = model_cart.loss(
                        pred_raw, pred, label, raw_boxes[..., :4]
                    )
                    total_loss = Det_loss + FFT_loss
                    total_loss_b, FFT_loss_b, Det_loss_b, box_loss_b, conf_loss_b, category_loss_b = (
                        total_loss.cpu().detach(),
                        FFT_loss.cpu().detach(),
                        Det_loss.cpu().detach(),
                        box_loss.cpu().detach(),
                        conf_loss.cpu().detach(),
                        category_loss.cpu().detach(),
                    )
                    
                    total_losstest.append(total_loss_b)
                    FFT_losstest.append(FFT_loss_b)
                    Det_losstest.append(Det_loss_b)
                    box_losstest.append(box_loss_b)
                    conf_losstest.append(conf_loss_b)
                    category_losstest.append(category_loss_b)
                    raw_boxes = raw_boxes.cpu().numpy()
                    pred = pred.cpu().detach().numpy()
                    for batch_id in range(raw_boxes.shape[0]):
                        raw_boxes_frame = raw_boxes[batch_id]
                        pred_frame = pred[batch_id]
                        predicitons = yoloheadToPredictions2D(
                            pred_frame,
                            conf_threshold=config_model["confidence_threshold"],
                        )
                        nms_pred = nms2DOverClass(
                            predicitons,
                            config_model["nms_iou3d_threshold"],
                            config_model["input_shape"],
                            sigma=0.3,
                            method="nms",
                        )
                        mean_ap, ap_all_class = mAP.mAP2D(
                            nms_pred,
                            raw_boxes_frame,
                            config_model["input_shape"],
                            ap_all_class,
                            tp_iou_threshold=config_model["mAP_iou3d_threshold"],
                        )
                        mean_ap_test += mean_ap
            for ap_class_i in ap_all_class:
                if len(ap_class_i) == 0:
                    class_ap = 0.0
                else:
                    class_ap = np.mean(ap_class_i)
                ap_all_class_test.append(class_ap)
            mean_ap_test = np.mean(ap_all_class_test)
            print("-------> ap: %.6f" % mean_ap_test)

            writer.add_scalar("ap/MAP", mean_ap_test, global_step=epoch)
            writer.add_scalar("ap/ap_person", ap_all_class_test[0], global_step=epoch)
            writer.add_scalar("ap/ap_bicycle", ap_all_class_test[1], global_step=epoch)
            writer.add_scalar("ap/ap_car", ap_all_class_test[2], global_step=epoch)
            writer.add_scalar("ap/ap_motorcycle", ap_all_class_test[3], global_step=epoch)
            writer.add_scalar("ap/ap_bus", ap_all_class_test[4], global_step=epoch)
            writer.add_scalar("ap/ap_truck", ap_all_class_test[5], global_step=epoch)
            ### NOTE: validate loss ###
            writer.add_scalar("validate_loss/total_loss", np.mean(total_losstest), global_step=epoch)
            writer.add_scalar("validate_loss/FFT_loss", np.mean(FFT_losstest), global_step=epoch)
            writer.add_scalar("validate_loss/Det_loss", np.mean(Det_losstest), global_step=epoch)
            writer.add_scalar("validate_loss/box_loss", np.mean(box_losstest), global_step=epoch)
            writer.add_scalar("validate_loss/conf_loss", np.mean(conf_losstest), global_step=epoch)
            writer.add_scalar("validate_loss/category_loss", np.mean(category_losstest), global_step=epoch)
            writer.flush()

            ckpt_file_path = os.path.join(log_specific_dir, "epoch{:02d}_Tloss_{:.4f}_MAP_{:.4f}.pth".format(
            epoch,total_loss_r,mean_ap_test))
            ckpt_file_best_path = os.path.join(log_specific_dir, "best.pth")

            checkpoint = {}
            checkpoint["net_state_dict"] = model_cart.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["scheduler"] = scheduler.state_dict()
            checkpoint["epoch"] = epoch
            checkpoint["global_step"] = step
            checkpoint["MAP"] = mean_ap_test

            with open(ckpt_file_path, "wb") as f:
                torch.save(checkpoint, f)
            print("Saving checkpoint to {}".format(ckpt_file_path))
            if mean_ap_test > best_val:
                best_val = mean_ap_test
                with open(ckpt_file_best_path, "wb") as f:
                    torch.save(checkpoint, f)
            print(
                "Current model is the best model at present, and it is saved to {}".format(
                    ckpt_file_best_path
                )
            )
            with open(log_name, "a") as f:
                f.write(
                    "{}  {:.6f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f} {:.4f} \n".format(
                        epoch,
                        scheduler.get_last_lr()[0],
                        Ttotal_loss / len(train_loader.dataset),
                        TFFT_loss / len(train_loader.dataset),
                        TDet_loss / len(train_loader.dataset),
                        np.mean(total_losstest),
                        np.mean(FFT_losstest),
                        np.mean(Det_losstest),
                        mean_ap_test,
                        ap_all_class_test[0],
                        ap_all_class_test[1],
                        ap_all_class_test[2],
                        ap_all_class_test[3],
                        ap_all_class_test[4],
                        ap_all_class_test[5]
                    )
                )

        if dist_utils.get_world_size() > 1 and dist_utils.get_world_size() > 0:
            torch.distributed.barrier()
        scheduler.step()


def get_parse():
    parser = argparse.ArgumentParser(description="Args for segmentation model.")
    parser.add_argument("--num-gpus", type=int, default=1, help="The number of gpus.")
    parser.add_argument(
        "--num-machines", type=int, default=1, help="The number of machines."
    )
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="The rank of current machine."
    )
    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend.",
    )
    parser.add_argument(
        "--backbone_resume_from",
        type=str,
        default=False,
        help="The number of machines.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,#'logs/RA_map_supervised_alpha=100-1_coplex_Angle_FFT_serial_attention_FPN_RD-b_6/ckpt/epoch62_Tloss_1.9024_MAP_0.4280.pth',
        type=str,
        help="Path to the .pth model checkpoint to resume training",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    print("Command Line Args: ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
