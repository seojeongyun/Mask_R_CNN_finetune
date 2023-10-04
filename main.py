# from engine.engine_mobilenet import Trainer
from engine.engine_rcnn import Trainer

if __name__ == '__main__':
    trainer = Trainer(epochs=10)
    trainer.train_model()