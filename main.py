import os
import json
import torch
import argparse
from datetime import datetime
from utils import ADFormer_config
from utils import utils
from utils.ADFormer_dataset import ADFormerDataset
from model.ADFormer import ADFormer
from utils.ADFormer_trainer import ADFormerTrainer


def run_model(model_name, mode, start_time, test_dir=None):

    if mode == 'train':
        args = ADFormer_config.get_args()
    elif mode == 'test':
        with open(os.path.join(test_dir, "args.json"), "r") as f:
            saved_args = json.load(f)
            args = argparse.Namespace(**saved_args)

    args.exp_id = start_time
    args.log_dir = utils.get_log_dir(model_name, args.dataset_name, args.exp_id)
    
    if mode == 'train':
        with open(os.path.join(args.log_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    logger = utils.get_logger(args.log_dir)
    logger.info(f'Start Experiment at {args.exp_id}, dataset: {args.dataset_name}, model: {model_name}, mode: {mode}')
    logger.info(f'Experiment Configuration: {vars(args)}')

    dataset = ADFormerDataset(args, logger)
    train_dataloader, val_dataloader, test_dataloader = dataset.get_data()
    dataset_feature = dataset.get_dataset_feature()

    if model_name == 'ADFormer':
        model = ADFormer(args, dataset_feature, logger)
    
    trainer = ADFormerTrainer(args, model, logger)
    if mode == 'train':
        trainer.train(train_dataloader, val_dataloader)
        model.load_state_dict(torch.load(os.path.join(args.log_dir, 'model_best_state.pth')))
        trainer.evaluate(test_dataloader)
    elif mode == 'test':
        best_model_path = os.path.join(test_dir, 'model_best_state.pth')
        logger.info(f'Loading trained model from {best_model_path}')
        model.load_state_dict(torch.load(best_model_path))
        trainer.evaluate(test_dataloader)


if __name__ == '__main__':

    start_time = datetime.now().strftime("%m-%d_%H-%M")
    run_model(
        model_name='ADFormer', 
        mode='train',
        start_time=start_time,
        test_dir=None,
    )
        
    
    
    