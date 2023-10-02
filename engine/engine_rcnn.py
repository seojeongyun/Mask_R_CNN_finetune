import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
from PIL import Image
from dataload.dataloader import PennFudanDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


cudnn.benchmark = True
plt.ion()   # interactive mode




class Trainer():
    def __init__(self, epochs=25, num_images=6):
        self.save_path = '../ckpt/last_ckpt.pt'
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #
        self.epochs = epochs
        self.num_images = num_images
        #
        self.loader = PennFudanDataset()
        #
        self.model = self.get_model()
        #
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        #
        self.predict = self.model_predict()
        #
        self.val_loss = []
        self.train_loss = []
        self.num = []

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def get_optimizer(self):
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optimizer_ft

    def get_scheduler(self):
        exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        return exp_lr_scheduler

    def get_model(self):
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
        since = time.time()

        # Create a temporary directory to save training checkpoints
        best_model_params_path = self.save_path

        torch.save(self.model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(self.epochs):
            self.num.append(epoch)
            print(f'Epoch {epoch}/{self.epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.loader.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.loader.len[phase]
                epoch_acc = running_corrects.double() / self.loader.len[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    self.val_loss.append(epoch_loss)
                else:
                    self.train_loss.append(epoch_loss)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(torch.load(best_model_params_path))
        return self.model

    def visualize_model(self):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.loader.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(self.num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.loader.cls_name[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == self.num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)

    def model_predict(self):
        was_training = self.model.training
        self.model.eval()

        img = Image.open('/storage/jysuh/hymenoptera_data/val/ants/94999827_36895faade.jpg')
        img = self.loader.transform['val'](img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, preds = torch.max(outputs, 1)

            ax = plt.subplot(2, 2, 1)
            ax.axis('off')
            ax.set_title(f'Predicted: {self.loader.cls_name[preds[0]]}')
            imshow(img.cpu().data[0])

            self.model.train(mode=was_training)

    def print_loss(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.xlabel('Epoch')
        plt.ylabel('train_Loss')
        plt.plot(self.num, self.train_loss)
        plt.subplot(1, 2, 2)
        plt.xlabel('Epoch')
        plt.ylabel('val_Loss')
        plt.plot(self.num, self.val_loss)
        plt.show()