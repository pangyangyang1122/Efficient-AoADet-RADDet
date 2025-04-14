# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import shutil
from tabulate import tabulate
import sys
sys.path.append('/media/sqpang/新加卷/rada_target_detection/code/FFT+Detection/')
from util import helper
from util import drawer
from util import loader
import pandas as pd
import time

import torch.optim
from dataset.radar_dataset import RararDataset

import argparse
import os

import numpy as np
from model.model import RADDet,FPNDet
from dataset.radar_dataset_3d import RararDataset3D
from torch.utils.data import DataLoader
from metrics import mAP
from model.model_cart import RADDetCart
from model.yolo_loss import yoloheadToPredictions2D
import copy
from model.yolo_head import decodeYolo, yoloheadToPredictions, nms
from dataset.radar_dataset_plot import RararDatasetEvaluate
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

def cutImage(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1540:1750, :]
    part_2 = image[:, 2970:3550, :]
    part_3 = image[:, 4370:5400, :]
    part_4 = image[:, 6200:6850, :]
    new_img = np.concatenate([part_4, part_1, part_2, part_3], axis=1)
    cv2.imwrite(image_name, new_img)


def cutImage3Axes(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1780:2000, :]
    part_2 = image[:, 3800:4350, :]
    part_3 = image[:, 5950:6620, :]
    new_img = np.concatenate([part_3, part_1, part_2], axis=1)
    cv2.imwrite(image_name, new_img)


### NOTE: define testing step for RAD Boxes Model ###
def test_step(config_model, model, test_dataloader, num_classes, input_size,
              anchor_boxes, map_iou_threshold_list):
    mean_ap_test_all = []
    ap_all_class_test_all = []
    ap_all_class_all = []
    for i in range(len(map_iou_threshold_list)):
        mean_ap_test_all.append(0.0)
        ap_all_class_test_all.append([])
        ap_all_class = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        ap_all_class_all.append(ap_all_class)
    print("Start evaluating RAD Boxes on the entire dataset, it might take a while...")
    # pbar = tqdm(total=int(data_generator.total_test_batches))
    for data, label, raw_boxes in test_dataloader:
        data = data.to(device)
        label = label.to(device)
        raw_boxes = raw_boxes.to(device)

        _, feature = model(data)
        pred_raw, pred = decodeYolo(feature,
                                    input_size=input_size,
                                    anchor_boxes=anchor_boxes,
                                    scale=config_model["yolohead_xyz_scales"][0])
        pred = pred.cpu().detach().numpy()
        raw_boxes = raw_boxes.cpu().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame = pred[batch_id]
            predicitons = yoloheadToPredictions(pred_frame,
                                                conf_threshold=config_model["confidence_threshold"])
            nms_pred = nms(predicitons, config_model["nms_iou3d_threshold"],
                           config_model["input_shape"], sigma=0.3, method="nms")
            for j in range(len(map_iou_threshold_list)):
                map_iou_threshold = map_iou_threshold_list[j]
                mean_ap, ap_all_class_all[j] = mAP.mAP(nms_pred, raw_boxes_frame,
                                                       config_model["input_shape"],
                                                       ap_all_class_all[j],
                                                       tp_iou_threshold=map_iou_threshold)
                mean_ap_test_all[j] += mean_ap

    for iou_threshold_i in range(len(map_iou_threshold_list)):
        ap_all_class = ap_all_class_all[iou_threshold_i]
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_all[iou_threshold_i].append(class_ap)
        mean_ap_test_all[iou_threshold_i] = np.mean(ap_all_class_test_all[iou_threshold_i])
    return mean_ap_test_all, ap_all_class_test_all

def plot_pred_RAD(ARD_pred,config_data,RA_map,RAD_file_spec):
    save_dir = "/media/sqpang/新加卷/rada_target_detection/Data/RADDet_reprocessed/DEbug/test/RADpred_w_RA_mapSupervised_flipped"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, RAD_file_spec[0].split("/")[-1].replace("npy", "png"))
    
    RAD_pred = ARD_pred.permute(0,2,1,3)
    RAD_pred = RAD_pred.detach().cpu().numpy().squeeze() #去掉batch维
    RA_map = RA_map.detach().cpu().numpy().squeeze()
    
    ###将归一化模值恢复为原始模值####
    RAD_mag_max = config_data["RAD_mag_max"]
    RAD_mag_min = config_data["RAD_mag_min"]
    normal_RAD_mag = np.abs(RAD_pred)
    RAD_mag = normal_RAD_mag * (RAD_mag_max - RAD_mag_min) + RAD_mag_min
    RAD_complex = RAD_mag * np.exp(1j * np.angle(RAD_pred))
    RAD_complex = RAD_pred
    RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex,
                                                        power_order=2), target_axis=-1), scalar=10, log_10=True)
    RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex,
                                                            power_order=2), target_axis=-2), scalar=10, log_10=True)
    RA_label = helper.getLog(RA_map, scalar=10, log_10=True)

    
    RA_img = helper.norm2Image(RA)[..., :3]
    RD_img = helper.norm2Image(RD)[..., :3]
    RA_label = helper.norm2Image(RA_label)[..., :3]

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[4,4,1], wspace=0.1)  # width_ratios=[4, 1] 指定第二个子图的宽度为第一个的1/4，wspace=0.1 减少子图之间的空白

    # 第一个子图
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(RA_label)
    ax1.set_title('Range-Azimuth_label')
    ax1.set_xticks([0, 64, 128, 192, 255])
    ax1.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
    ax1.set_yticks([0, 64, 128, 192, 255])
    ax1.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax1.set_xlabel("angle (degrees)")
    ax1.set_ylabel("range (m)")

    # 第二个子图

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(RA_img)
    ax1.set_title('Range-Azimuth')
    ax1.set_xticks([0, 64, 128, 192, 255])
    ax1.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
    ax1.set_yticks([0, 64, 128, 192, 255])
    ax1.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax1.set_xlabel("angle (degrees)")
    ax1.set_ylabel("range (m)")

    # 第三个子图
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(RD_img)
    ax2.set_title('Range-Doppler')
    ax2.set_xticks([0, 16, 32, 48, 63])
    ax2.set_xticklabels([-13, -6.5, 0, 6.5, 13])
    ax2.set_yticks([0, 64, 128, 192, 255])
    ax2.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax2.set_xlabel("velocity (m/s)")
    ax2.set_ylabel("range (m)")
    plt.savefig(file_name)  # 保存图像
    plt.close(fig)  # 关闭当前图形，释放内存
    
## NOTE: define testing step for Cartesian Boxes Model ###
def test_step_cart(config_model, config_data, config_evaluate, model_cart, test_dataloader, num_classes, map_iou_threshold_list):
    mean_ap_test_all = []
    ap_all_class_test_all = []
    ap_all_class_all = []
    for i in range(len(map_iou_threshold_list)):
        mean_ap_test_all.append(0.0)
        ap_all_class_test_all.append([])
        ap_all_class = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        ap_all_class_all.append(ap_all_class)
    print("Start evaluating Cartesian Boxes on the entire dataset, it might take a while...")
    for data, label, raw_boxes, _, RAD_file_spec in test_dataloader:
        data = data.to(device)
        RA_map = label
        raw_boxes = raw_boxes.to(device)

        pred_raw, ARD_pred = model_cart(data)
        #######画出预测的RA，RD图以及标签图###############
        plot_pred_RAD(ARD_pred,config_data, RA_map, RAD_file_spec)
        ######################################
        pred = model_cart.decodeYolo(pred_raw)

        raw_boxes = raw_boxes.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame = pred[batch_id]
            predicitons = yoloheadToPredictions2D(pred_frame,
                                                  conf_threshold=0.05)
            nms_pred = helper.nms2D(predicitons,
                                    config_evaluate["nms_iou3d_threshold"],
                                    config_model["input_shape"], sigma=0.3, method="nms")
            for j in range(len(map_iou_threshold_list)):
                map_iou_threshold = map_iou_threshold_list[j]
                mean_ap, ap_all_class_all[j] = mAP.mAP2D(nms_pred, raw_boxes_frame, config_model["input_shape"],
                                                         ap_all_class_all[j],
                                                         tp_iou_threshold=map_iou_threshold)
                mean_ap_test_all[j] += mean_ap
    for iou_threshold_i in range(len(map_iou_threshold_list)):
        ap_all_class = ap_all_class_all[iou_threshold_i]
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_all[iou_threshold_i].append(class_ap)
        mean_ap_test_all[iou_threshold_i] = np.mean(ap_all_class_test_all[iou_threshold_i])
    return mean_ap_test_all, ap_all_class_test_all

def predictionPlots(config_data, if_evaluate_cart, model, model_cart, radar_dataset,
                    config_evaluate, config_model, input_size, anchor_boxes):
    """ Plot the predictions of all data in dataset """
    if if_evaluate_cart:
        fig, axes = drawer.prepareFigure(4, figsize=(80, 6))
    else:
        fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
    # colors = loader.randomColors(config_data["all_classes"])
    gt_colors = drawer.SetColorsGreen(len(config_data["all_classes"]))
    pred_colors = drawer.SetColorsRed(len(config_data["all_classes"]))

    image_save_dir = "./images/evaluate_plots/" + config_data["RAD_dir"]
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    else:
        shutil.rmtree(image_save_dir)
        os.makedirs(image_save_dir)
    print("Start plotting, it might take a while...")
    for RAD_file, data, label, label_cart, raw_boxes, raw_boxes_cart, \
            stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances in radar_dataset:
        data = data.to(device)
        feature_bk, feature = model(data)
        pred_raw, pred = decodeYolo(feature,
                                    input_size=input_size,
                                    anchor_boxes=anchor_boxes,
                                    scale=config_model["yolohead_xyz_scales"][0])
        pred = pred.cpu().detach().numpy()
        pred_frame = pred[0]
        predicitons = helper.yoloheadToPredictions(pred_frame, conf_threshold=0.5)
        nms_pred = helper.nmsOverClass(predicitons, config_evaluate["nms_iou3d_threshold"],
                                       config_model["input_shape"], sigma=0.3, method="nms")

        if if_evaluate_cart:
            pred_raw_cart = model_cart(data)
            pred_cart = model_cart.decodeYolo(pred_raw_cart)
            pred_cart = pred_cart.cpu().detach().numpy()
            pred_frame_cart = pred_cart[0]
            predicitons_cart = helper.yoloheadToPredictions2D(pred_frame_cart,
                                                              conf_threshold=0.5)
            nms_pred_cart = helper.nms2DOverClass(predicitons_cart, \
                                                  config_model["nms_iou3d_threshold"], \
                                                  config_model["input_shape"], \
                                                  sigma=0.3, method="nms")
        else:
            nms_pred_cart = None

        drawer.clearAxes(axes)
        drawer.drawRadarPredWithGt(stereo_left_image, RD_img, \
                                   RA_img, RA_cart_img, gt_instances, nms_pred, \
                                   config_data["all_classes"], gt_colors, pred_colors, axes, \
                                   radar_cart_nms=nms_pred_cart)
        dataID = os.path.splitext(os.path.basename(RAD_file))[0]
        drawer.saveFigure(image_save_dir, "%s.png" % (dataID))
        if if_evaluate_cart:
            cutImage(image_save_dir, "%s.png" % (dataID))
        else:
            cutImage3Axes(image_save_dir, "%s.png" % (dataID))


def main(args):
    # initialization
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    # env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(env_str)

    config = loader.readConfig(config_file_name="configFPN.json")
    config_data = config["DATA"]
    #####测试时，模型不用加载RAD数据############
    # config_data["RAD_supervised"] = False
    ############3333#######################33
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]
    RAD_dir = config_data['input_type'] 
    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="anchors.txt")
    num_classes = len(config_data["all_classes"])
    anchor_boxes_cart = loader.readAnchorBoxes(anchor_boxes_file="anchors_cartboxes.txt")

    anchor_boxes_bk = copy.deepcopy(anchor_boxes)
    anchor_boxes_cart_bk = copy.deepcopy(anchor_boxes_cart)

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = FPNDet(config_model, config_data, config_train, anchor_boxes)
    """
    # model.load_state_dict(torch.load(args.resume_from))  ######只评估笛卡尔坐标系
    """
    model_bk = copy.deepcopy(model.backbone)
    model_RADec = copy.deepcopy(model.RA_decoder)
    # model.to(device)

    model_cart = RADDetCart(config_model, config_data, config_train, anchor_boxes_cart, config_model["bk_output_size"],
                            device, backbone=model_bk, RA_decoder=model_RADec,)
    dict = torch.load(args.cart_resume_from)
    model_cart.load_state_dict(dict["net_state_dict"])
    model_cart.to(device)

    # test_dataset = RararDataset(config_data, config_train, config_model,
    #                             config_model["feature_out_shape"], anchor_boxes, dType="test", RADDir=RAD_dir)    # 2032
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=config_train["batch_size"]//args.num_gpus,
    #                          shuffle=False,
    #                          num_workers=4,
    #                          pin_memory=True,
    #                          persistent_workers=True)

    start_time = time.time()
    test_dataset_cart = RararDataset3D(config_data=config_data,
                                       config_train=config_train,
                                       config_model=config_model,
                                       headoutput_shape=config_data["headoutput_shape"],
                                       anchors=anchor_boxes,
                                       anchors_cart=anchor_boxes_cart,
                                       cart_shape=config_data["cart_shape"],
                                       dType="test",
                                       Input_type=RAD_dir)

    test_loader_cart = DataLoader(test_dataset_cart,
                                  batch_size=config_train["batch_size"]//args.num_gpus,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  persistent_workers=True)


    ### NOTE: RAD Boxes ckpt ###
    # logdir = os.path.join(config_evaluate["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
    #                       "-b_" + str(config_train["batch_size"]) +
    #                       "-lr_" + str(config_train["learningrate_init"]))
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)

    ### NOTE: Cartesian Boxes ckpt ###
    if_evaluate_cart = True
    logdir_cart = os.path.join(config_train["log_dir"], config["NAME"]+"_"+config_data["input_type"]+ "_cartesian")
    if not os.path.exists(logdir_cart):
        os.makedirs(logdir_cart)

    output_dir = os.path.join(logdir_cart, 'results_mAP')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)

    ### NOTE: evaluate RAD Boxes under different mAP_iou ###
    # all_mean_aps, all_ap_classes = test_step(config_model=config_model,
    #                                          model=model,
    #                                          test_dataloader=test_loader,
    #                                          num_classes=num_classes,
    #                                          input_size=input_size,
    #                                          anchor_boxes=anchor_boxes,
    #                                          map_iou_threshold_list=config_evaluate["mAP_iou3d_threshold"]
    #                                          )

    # all_mean_aps = np.array(all_mean_aps)
    # all_ap_classes = np.array(all_ap_classes)

    # table = []
    # row = []
    # for i in range(len(all_mean_aps)):
    #     if i == 0:
    #         row.append("mAP")
    #     row.append(all_mean_aps[i])
    # table.append(row)
    # row = []
    # for j in range(all_ap_classes.shape[1]):
    #     ap_current_class = all_ap_classes[:, j]
    #     for k in range(len(ap_current_class)):
    #         if k == 0:
    #             row.append(config_data["all_classes"][j])
    #         row.append(ap_current_class[k])
    #     table.append(row)
    #     row = []
    # headers = []
    # for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
    #     if ap_iou_i == 0:
    #         headers.append("AP name")
    #     headers.append("AP_%.2f"%(ap_iou_i))
    # print("==================== RAD Boxes AP ========================")
    # print(tabulate(table, headers=headers))
    # print("==========================================================")
    # filename = os.path.join(output_dir, config_data["RAD_dir"]+"-RAD Boxes AP.csv")
    # df = pd.DataFrame(table, columns=['metric']+[ap for ap in headers])
    # df.to_csv(filename, index=False, float_format="%.3f")

    ### NOTE: evaluate Cart Boxes under different mAP_iou ###
    if if_evaluate_cart:
        all_mean_aps, all_ap_classes = test_step_cart(config_model=config_model,
                                                      config_data=config_data,
                                                      config_evaluate=config_evaluate,
                                                      model_cart=model_cart,
                                                      test_dataloader=test_loader_cart,
                                                      num_classes=num_classes,
                                                      map_iou_threshold_list=config_evaluate["mAP_iou3d_threshold"]
                                                      )
        total_time = time.time() - start_time
        average_time_per_sample = total_time / len(test_loader_cart.dataset)
        print ("Average time per sample:",average_time_per_sample)
        
        all_mean_aps = np.array(all_mean_aps)
        all_ap_classes = np.array(all_ap_classes)

        table = []
        row = []
        for i in range(len(all_mean_aps)):
            if i == 0:
                row.append("mAP")
            row.append(all_mean_aps[i])
        table.append(row)
        row = []
        for j in range(all_ap_classes.shape[1]):
            ap_current_class = all_ap_classes[:, j]
            for k in range(len(ap_current_class)):
                if k == 0:
                    row.append(config_data["all_classes"][j])
                row.append(ap_current_class[k])
            table.append(row)
            row = []
        headers = []
        for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
            if ap_iou_i == 0:
                headers.append("AP name")
            headers.append("AP_%.2f" % (ap_iou_i))
        print("================= Cartesian Boxes AP =====================")
        print(tabulate(table, headers=headers))
        print("==========================================================")
        filename = os.path.join(output_dir, config_data["input_type"] + "-Cartesian Boxes AP.csv")
        df = pd.DataFrame(table, columns=['metric'] + [ap for ap in headers])
        df.to_csv(filename, index=False, float_format="%.3f")

    # radar_dataset_plot = RararDatasetEvaluate(
    #     config_data=config_data,
    #     config_train=config_train,
    #     config_model=config_model,
    #     config_radar=config_radar,
    #     headoutput_shape=config_data["headoutput_shape"],
    #     anchors=anchor_boxes_bk,
    #     anchors_cart=anchor_boxes_cart_bk,
    #     cart_shape=config_data["cart_shape"],
    #     dType="test",
    #     RADDir=RAD_dir
    # )

    # radar_dataset_plot_loader = DataLoader(radar_dataset_plot,
    #                                        batch_size=1,
    #                                        shuffle=True,
    #                                        num_workers=4,
    #                                        pin_memory=True,
    #                                        persistent_workers=True)

    # NOTE: plot the predictions on the entire dataset ###
    # predictionPlots(config_data, True, model, model_cart,
    #                 radar_dataset=iter(radar_dataset_plot),
    #                 config_evaluate=config_evaluate,
    #                 config_model=config_model,
    #                 input_size=input_size,
    #                 anchor_boxes=anchor_boxes,
    #                 )


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="The number of gpus.")
    parser.add_argument("--num-machines", type=int,
                        default=1,
                        help="The number of machines.")
    parser.add_argument("--machine-rank", type=int,
                        default=0,
                        help="The rank of current machine.")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist_url", type=str,
                        default="tcp://127.0.0.1:{}".format(port),
                        help="initialization URL for pytorch distributed backend.")
    parser.add_argument("--resume_from", type=str,
                        default= "/media/sqpang/新加卷/rada_target_detection/code/FFT+Detection/logs/RA_map_supervised_alpha=100-1_coplex_Angle_FFT_serial_attention_FPNX2_RD-b_6/ckpt/epoch26_Tloss_3.6468_MAP_0.4527.pth",
                        help="The number of machines.")
    parser.add_argument("--cart_resume_from", type=str,
                        default="./logs/coplex_Angle_FFT_serial_attention_FPN_RD_cartesian/ckpt/best.pth",
                        help="The number of machines.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    # RAD_dir = 'RAD', 'RAD4', 'RAD_Sim8'
    main(args)
    # main(args, RAD_dir='RAD4')
    # main(args, RAD_dir='RAD_Sim8')

