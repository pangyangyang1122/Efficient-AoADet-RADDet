from torch.utils.data import Dataset
import numpy as np
import os, glob
import util.loader as loader
import util.helper as helper
from torchvision.transforms import ToTensor
import torch
import mkl_fft
from scipy import signal


class RararDataset3D(Dataset):
    """load cartesian data"""

    def __init__(
        self,
        config_data,
        config_train,
        config_model,
        headoutput_shape,
        anchors,
        transformer=ToTensor(),
        anchors_cart=None,
        cart_shape=None,
        dType="train",
        Input_type="RD",
    ):
        super(RararDataset3D, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.Input_type = Input_type
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RD_sequences_train = self.readSequences(mode="train")
        self.RD_sequences_test = self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.RD_sequences_train, self.RD_sequences_validate = self.splitTrain(
            self.RD_sequences_train
        )
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (
            self.config_train["epochs"] * len(self.RD_sequences_train)
        ) // self.batch_size
        self.total_test_batches = len(self.RD_sequences_test) // self.batch_size
        self.total_validate_batches = (
            len(self.RD_sequences_validate) // self.batch_size
        )
        self.dtype = dType
        self.transform = transformer
        self.RAD_supervised = config_data["RAD_supervised"]
        self.RD_mag_max = config_data["RD_mag_max"]
        self.RD_mag_min = config_data["RD_mag_min"]
        self.RAD_mag_max = config_data["RD_mag_max"]
        self.RAD_mag_min = config_data["RD_mag_min"]

    def __len__(self):
        if self.dtype == "train":
            return len(self.RD_sequences_train)
        elif self.dtype == "validate":
            return len(self.RD_sequences_validate)
        elif self.dtype == "test":
            return len(self.RD_sequences_test)
        else:
            raise ValueError("This type of dataset does not exist.")

    def __getitem__(self, index):
        if self.dtype == "train":
            return self.trainDataCart(index)
        elif self.dtype == "validate":
            return self.validateDataCart(index)
        elif self.dtype == "test":
            return self.testDataCart(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def trainDataCart(self, index):
        """Generate train data with batch size"""
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")

        has_label = False
        while not has_label:
            RD_filename = self.RD_sequences_train[index]
            RD_complex = loader.readRAD(RD_filename)
            if RD_complex is None:
                raise ValueError("RD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RD_mag = np.abs(RD_complex)
            normal_RD_mag = (RD_mag - self.RD_mag_min) / (self.RD_mag_max - self.RD_mag_min)
            normal_RD = normal_RD_mag * np.exp(1j * np.angle(RD_complex))

            ### load ground truth instances ###
            if self.RAD_supervised:
                #########使用加载的方式获得RAD#####
                # RAD_filename= loader.getRADFromRD(RD_filename, self.config_data["train_set_dir"], self.Input_type)
                # RAD_complex = np.load(RAD_filename)
                # RAD_mag = np.abs(RAD_complex)
                # normal_RAD_mag = (RAD_mag - self.RAD_mag_min) / (self.RAD_mag_max - self.RAD_mag_min)
                # normal_RAD = normal_RAD_mag * np.exp(1j * np.angle(RAD_complex))
                ############使用计算的方式获得RAD########
                normal_RAD_spectrums = self.get_RAD(normal_RD)
                normal_RA_map = helper.getSumDim(helper.getMagnitude(normal_RAD_spectrums,power_order=2), target_axis=-1)
                

            gt_filename, RAD_file_spec = loader.gtfileFromRADfile(
                RD_filename, self.config_data["train_set_dir"], self.Input_type
            )
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.RAD_supervised:
                    return (
                        self.transform(normal_RD),
                        self.transform(normal_RA_map),
                        torch.tensor(gt_labels, dtype=torch.float32),
                        torch.tensor(raw_boxes, dtype=torch.float32),
                    )
                else:
                    return (
                        self.transform(normal_RD),
                        torch.tensor(gt_labels, dtype=torch.float32),
                        torch.tensor(raw_boxes, dtype=torch.float32),
                    )

    def testDataCart(self, index):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RD_filename = self.RD_sequences_test[index]
            RD_complex = loader.readRAD(RD_filename)
            if RD_complex is None:
                raise ValueError("RD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RD_mag = np.abs(RD_complex)
            normal_RD_mag = (RD_mag - self.RD_mag_min) / (self.RD_mag_max - self.RD_mag_min)
            normal_RD = normal_RD_mag * np.exp(1j * np.angle(RD_complex))

            ### load ground truth instances ###
            if self.RAD_supervised:
                #########使用加载的方式获得RAD#####
                # RAD_filename= loader.getRADFromRD(RD_filename, self.config_data["train_set_dir"], self.Input_type)
                # RAD_complex = np.load(RAD_filename)
                # RAD_mag = np.abs(RAD_complex)
                # normal_RAD_mag = (RAD_mag - self.RAD_mag_min) / (self.RAD_mag_max - self.RAD_mag_min)
                # normal_RAD = normal_RAD_mag * np.exp(1j * np.angle(RAD_complex))
                ############使用计算的方式获得RAD########
                # normal_RAD_spectrums = self.get_RAD(normal_RD)  #使用的是归一化的RAD监督
                normal_RAD_spectrums = self.get_RAD(RD_complex)   #使用不归一化RAD,画出的图像才比较像
                normal_RA_map = helper.getSumDim(helper.getMagnitude(normal_RAD_spectrums,power_order=2), target_axis=-1)

            gt_filename,RAD_file_spec = loader.gtfileFromRADfile(
                RD_filename, self.config_data["test_set_dir"], self.Input_type
            )
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.RAD_supervised:
                    return (
                        self.transform(normal_RD),
                        self.transform(normal_RA_map),
                        torch.tensor(gt_labels, dtype=torch.float32),
                        torch.tensor(raw_boxes, dtype=torch.float32),
                        RAD_file_spec
                    )
                else:
                    return (
                        self.transform(normal_RD),
                        torch.tensor(gt_labels, dtype=torch.float32),
                        torch.tensor(raw_boxes, dtype=torch.float32),
                        RAD_file_spec
                    )

    def validateDataCart(self, index):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RD_filename = self.RD_sequences_validate[index]
            RD_complex = loader.readRAD(RD_filename)
            if RD_complex is None:
                raise ValueError("RD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RD_mag = np.abs(RD_complex)
            normal_RD_mag = (RD_mag - self.RD_mag_min) / (self.RD_mag_max - self.RD_mag_min)
            normal_RD = normal_RD_mag * np.exp(1j * np.angle(RD_complex))

            ### load ground truth instances ###
            if self.RAD_supervised:
                #########使用加载的方式获得RAD#####
                # RAD_filename= loader.getRADFromRD(RD_filename, self.config_data["train_set_dir"], self.Input_type)
                # RAD_complex = np.load(RAD_filename)
                # RAD_mag = np.abs(RAD_complex)
                # normal_RAD_mag = (RAD_mag - self.RAD_mag_min) / (self.RAD_mag_max - self.RAD_mag_min)
                # normal_RAD = normal_RAD_mag * np.exp(1j * np.angle(RAD_complex))
                ############使用计算的方式获得RAD########
                normal_RAD_spectrums = self.get_RAD(normal_RD)
                normal_RA_map = helper.getSumDim(helper.getMagnitude(normal_RAD_spectrums,power_order=2), target_axis=-1)

            gt_filename ,RAD_file_spec= loader.gtfileFromRADfile(
                RD_filename, self.config_data["train_set_dir"], self.Input_type
            )
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.RAD_supervised:
                    return (
                        self.transform(normal_RD),
                        self.transform(normal_RA_map),
                        torch.tensor(gt_labels, dtype=torch.float32),
                        torch.tensor(raw_boxes, dtype=torch.float32),
                    )
                else:
                    return (
                        self.transform(normal_RD),
                        torch.tensor(gt_labels, dtype=torch.float32),
                        torch.tensor(raw_boxes, dtype=torch.float32),
                    )

    def get_RAD(self,RD_spectrums):

        # zero padding 256是FFT后要达到的角度维
        RD = np.zeros((RD_spectrums.shape[0], RD_spectrums.shape[1], 256), dtype=RD_spectrums.dtype)
        RD[:,:,0:RD_spectrums.shape[2]] = RD_spectrums

        # RAD_spectrums = mkl_fft.fft(np.multiply(RD,self.azimuth_fft_coef), self.numSampleAzimuth, axis=2) 
        RAD_spectrums = mkl_fft.fft(RD, 256, axis=2)
        RAD_spectrums = np.fft.fftshift(RAD_spectrums, axes=2)
        
        # scale the output for power specturm estimation
        # RAD_spectrums = RAD_spectrums/(np.sqrt(2*np.pi*self.numSampleAzimuth))

        # reshape RAD: range-azimuth-chirp=256x256x64
        RAD_spectrums = RAD_spectrums.transpose((0,2,1))  

        # RADDet did this ???????? yes
        RAD_spectrums = np.flip(RAD_spectrums, axis=1)  

        return RAD_spectrums
    
    def encodeToCartBoxesLabels(self, gt_instances):
        """Transfer ground truth instances into Detection Head format"""
        raw_boxes_xywh = np.zeros((self.config_data["max_boxes_per_frame"], 5))
        ### initialize gronud truth labels as np.zeros ###
        gt_labels = np.zeros(
            list(self.cart_shape[1:3])
            + [len(self.anchor_boxes_cart)]
            + [len(self.config_data["all_classes"]) + 5]
        )
        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xywh = gt_instances["cart_boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i <= self.config_data["max_boxes_per_frame"]:
                raw_boxes_xywh[i, :4] = box_xywh
                raw_boxes_xywh[i, 4] = class_id
            class_onehot = helper.smoothOnehot(
                class_id, len(self.config_data["all_classes"])
            )
            exist_positive = False
            grid_strid = self.cart_grid_strides
            anchors = self.anchor_boxes_cart
            box_xywh_scaled = box_xywh[np.newaxis, :].astype(np.float32)
            box_xywh_scaled[:, :2] /= grid_strid
            anchors_xywh = np.zeros([len(anchors), 4])
            anchors_xywh[:, :2] = np.floor(box_xywh_scaled[:, :2]) + 0.5
            anchors_xywh[:, 2:] = anchors.astype(np.float32)  # box的w,h就是anchor的长和宽

            iou_scaled = helper.iou2d(box_xywh_scaled, anchors_xywh)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(np.squeeze(box_xywh_scaled)[:2]).astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, iou_mask, 0:4] = box_xywh
                gt_labels[xind, yind, iou_mask, 4:5] = 1.0
                gt_labels[xind, yind, iou_mask, 5:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                iou_mask = iou_scaled == iou_scaled.max()

                if np.any(iou_mask):
                    xind, yind = np.floor(np.squeeze(box_xywh_scaled)[:2]).astype(
                        np.int32
                    )
                    ### TODO: consider changing the box to raw yolohead output format ###
                    gt_labels[xind, yind, iou_mask, 0:4] = box_xywh
                    gt_labels[xind, yind, iou_mask, 4:5] = 1.0
                    gt_labels[xind, yind, iou_mask, 5:] = class_onehot

        has_label = False
        if gt_labels.max() != 0:
            has_label = True
        gt_labels = np.where(gt_labels == 0, 1e-16, gt_labels)
        return gt_labels, has_label, raw_boxes_xywh

    def getGridStrides(
        self,
    ):
        """Get grid strides"""
        strides = np.array(self.config_model["input_shape"])[:3] / np.array(
            self.headoutput_shape[1:4]
        )
        return np.array(strides).astype(np.float32)

    def getCartGridStrides(
        self,
    ):
        """Get grid strides"""
        if self.cart_shape is not None:
            cart_output_shape = [
                int(self.config_model["input_shape"][0]),
                int(2 * self.config_model["input_shape"][0]),
            ]
            strides = np.array(cart_output_shape) / np.array(self.cart_shape[1:3])
            return np.array(strides).astype(np.float32)
        else:
            return None

    def readSequences(self, mode):
        """Read sequences from train/test directories."""
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(
                os.path.join(
                    self.config_data["train_set_dir"], f"{self.Input_type}/*/*.npy"
                )
            )
        elif mode == "test":
            sequences = glob.glob(
                os.path.join(self.config_data["test_set_dir"], f"{self.Input_type}/*/*.npy")
            )
        else:
            raise ValueError(f"{mode} type does not exist.")
        if len(sequences) == 0:
            raise ValueError(
                "Cannot read data from either train or test directory, \
                        Please double-check the data path or the data format."
            )
        return sequences

    def splitTrain(self, train_sequences):
        """Split train set to train and validate"""
        total_num = len(train_sequences)
        validate_num = int(0.1 * total_num)
        if self.config_train["if_validate"]:
            return (
                train_sequences[: total_num - validate_num],
                train_sequences[total_num - validate_num :],
            )
        else:
            return train_sequences, train_sequences[total_num - validate_num :]
