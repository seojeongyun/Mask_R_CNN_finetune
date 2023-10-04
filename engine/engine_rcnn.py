import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import matplotlib.pyplot as plt

from dataload.dataloader import PennFudanDataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from coco.engine import train_one_epoch, evaluate
from coco import utils

cudnn.benchmark = True
plt.ion()   # interactive mode




class Trainer():
    def __init__(self, epochs=10):
        self.save_path = '../ckpt/last_ckpt.pt'
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #
        self.epochs = epochs

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def get_model(self, num_classes):
        # COCO로 미리 학습된 모델 읽기
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # 분류기를 새로운 것으로 교체하는데, num_classes는 사용자가 정의합니다
        num_classes = 2  # 1 클래스(사람) + 배경
        # 분류기에서 사용할 입력 특징의 차원 정보를 얻습니다
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # 미리 학습된 모델의 머리 부분을 새로운 것으로 교체합니다
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model.roi_heads.box_predictor

    def train_model(self):
        # 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 합니다
        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

        # 우리 데이터셋은 두 개의 클래스만 가집니다 - 배경과 사람
        num_classes = 2
        # 데이터셋과 정의된 변환들을 사용합니다
        dataset = PennFudanDataset(root='/storage/jysuh/PennFudanPed', task='train')
        dataset_test = PennFudanDataset(root='/storage/jysuh/PennFudanPed', task='not_train')

        # 데이터셋을 학습용과 테스트용으로 나눕니다(역자주: 여기서는 전체의 50개를 테스트에, 나머지를 학습에 사용합니다)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # 데이터 로더를 학습용과 검증용으로 정의합니다
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        # 도움 함수를 이용해 모델을 가져옵니다
        model = self.get_model(num_classes)

        # 모델을 GPU나 CPU로 옮깁니다
        model.to(device)

        # 옵티마이저(Optimizer)를 만듭니다
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # 학습률 스케쥴러를 만듭니다
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)


        for epoch in range(self.epochs):
            # 1 에포크동안 학습하고, 10회 마다 출력합니다
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # 학습률을 업데이트 합니다
            lr_scheduler.step()
            # 테스트 데이터셋에서 평가를 합니다
            evaluate(model, data_loader_test, device=device)

        print("That's it!")