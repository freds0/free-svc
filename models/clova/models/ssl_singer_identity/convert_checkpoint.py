import torch
import os
import yaml
import argparse
import sys
from singer_identity.model import load_model
from singer_identity.trainer import SSLTrainer
from singer_identity.trainer_byol import BYOL
from singer_identity.model import IdentityEncoder

def convert_model(checkpoint_path, config_path, directory_to_convert=None, num_trials=50):

    with open(config_path, 'r') as f:
        trainer_dict = yaml.safe_load(f)
    model_classpath = trainer_dict['model']['class_path']
    trainer_dict = trainer_dict['model']['init_args']

    ckpt_file = checkpoint_path

    if 'byol' in model_classpath.lower():
        byol = True
    else:
        byol = False

    modify_ckpt = False
    # Load model from checkpoint
    if byol:
        ckpt = torch.load(ckpt_file)
        projection = ckpt['hyper_parameters']['projection']
        if not isinstance(projection, dict):
            projection_dict = trainer_dict['projection']
            ckpt['hyper_parameters']['projection'] = projection_dict
            modify_ckpt = True
        predictor = ckpt['hyper_parameters']['predictor']
        if not isinstance(predictor, dict):
            predictor_dict = trainer_dict['predictor']
            ckpt['hyper_parameters']['predictor'] = predictor_dict
            modify_ckpt = True

        if modify_ckpt:
            ckpt_file = ckpt_file[:-5] + '_modified.ckpt'
            print("-------------------------------------")
            print("Saving modified checkpoint to", ckpt_file)
            print("-------------------------------------")
            torch.save(ckpt, ckpt_file)
        model = BYOL.load_from_checkpoint(ckpt_file)
    else:
        SSLTrainer(**trainer_dict)
        model = SSLTrainer.load_from_checkpoint(ckpt_file)

    feature_extractor_args = trainer_dict['feature_extractor']
    encoder_args = trainer_dict['backbone']
    model_args = {'feature_extractor': feature_extractor_args, 'encoder': encoder_args}

    # If not provided, save the converted model in a subfolder of the config file directory
    # Subfolder name is "converted"
    # Inside the subfolder, save the feature extractor, encoder, and hyperparameters in another subfolder
    # Subfolder name is "converted_<ckpt_basename>"
    if directory_to_convert is None:
        directory_to_convert = os.path.dirname(ckpt_file)
        directory_to_convert = f"{directory_to_convert}/converted/{os.path.basename(checkpoint_path).split('.')[0]}"
        # Create the directory if it doesn't exist
    if not os.path.exists(directory_to_convert):
        os.makedirs(directory_to_convert)
        
        
    
    with open(f'{directory_to_convert}/feature_extractor.yaml', 'w') as f:
        yaml.dump(feature_extractor_args, f)
    with open(f'{directory_to_convert}/encoder.yaml', 'w') as f:
        yaml.dump(encoder_args, f)
    with open(f'{directory_to_convert}/hyperparams.yaml', 'w') as f:
        yaml.dump(model_args, f)

    feature_extractor_args = yaml.safe_load(open(f'{directory_to_convert}/feature_extractor.yaml', 'r'))
    encoder_args = yaml.safe_load(open(f'{directory_to_convert}/encoder.yaml', 'r'))
    hyperparams = yaml.safe_load(open(f'{directory_to_convert}/hyperparams.yaml', 'r'))
    
    if byol:
        model.feature_extractor = model.module.encoder.feature_extractor
        model.encoder = model.module.encoder.encoder

    new_model = IdentityEncoder(**model_args)
    new_model.feature_extractor = model.feature_extractor
    new_model.encoder = model.encoder

    # Check that new_model and model have the same output

    model.eval()
    new_model.eval()
    for i in range(num_trials):
        x = torch.randn(2, 176000)
        assert torch.allclose(model.encoder(model.feature_extractor(x)), new_model(x))

    torch.save(new_model.state_dict(), f'{directory_to_convert}/model.pt')

    # directory_to_convert is /.../parent/folder_name
    # Add / to the end of directory_to_convert if it doesn't exist
    if directory_to_convert[-1] != '/':
        directory_to_convert += '/'
    print("directory_to_convert:", directory_to_convert)
    # Get folder_name
    model_name = os.path.basename(os.path.dirname(directory_to_convert))
    print("model_name:", model_name)
    # Get parent folder
    source = os.path.dirname(os.path.dirname(directory_to_convert))
    # If source is empty, set it to current directory
    if source == '':
        source = '.'
    
    print("source:", source)

    model = load_model(model=model_name, source=source)
    model.eval()
    for i in range(num_trials):
        x = torch.randn(2, 176000)
        assert torch.allclose(model(x), new_model(x))
    
    # Delete the temporary modified checkpoint file if it was modified
    if modify_ckpt:
        os.remove(ckpt_file)

    print("-------------------------------------")
    print("Model conversion complete!")
    print("-------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model checkpoint to another format.")
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint')
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    parser.add_argument('--output_dir', default=None, help='Directory to save the converted model. Default is a subfolder named "<ckpt_basename>" in the directory of the config file')
    parser.add_argument('--num_trials', default=20, help='Number of trials to run to check that the converted model is the same as the original model')
    args = parser.parse_args()

    convert_model(args.checkpoint, args.config, args.output_dir, args.num_trials)
