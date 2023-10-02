from engine.engine_mobilenet import Trainer

if __name__ == '__main__':
    trainer = Trainer(epochs=10)
    trainer.train_model()