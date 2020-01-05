from config import config
from loader.ImageBatcher import load_split_train_test
from models.backbone import bb_fc_model
import os

if __name__ == "__main__":
    model_list = ['resnet18', 'alexnet' ,'vgg16', 'squeezenet', 'densenet', 'inception', 'googlenet', 'shufflenet', 'mobilenet', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet']
    cur_path = os.getcwd()
    config = config().json_data
    data_path = os.path.join(cur_path, config["MODEL"]["DATA_LOAD_PATH"])
    model_save_path = os.path.join(cur_path, config["MODEL"]["MODEL_SAVE_PATH"])
    test_size = config["MODEL"]["VALIDATION_SIZE"]
    train_loader, test_loader = load_split_train_test(data_path, test_size)

    for model in model_list:
        backbone = bb_fc_model(model_name=model)
        backbone.train(train_loader, test_loader)
        backbone.save_model(file_path=model_save_path)
