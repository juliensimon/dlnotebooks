import os, argparse, pickle
import xgboost as xgb
import pandas as pd

print("XGBoost", xgb.__version__)

def load_dataset(path, sep):
    # Load dataset
    data = pd.read_csv(path, sep=sep)
    # Process dataset
    x = data.drop(['y_yes'], axis=1)
    y = data['y_yes']
    return x,y

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgb.model'))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
   
    args, _ = parser.parse_known_args()
    max_depth  = args.max_depth
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    x_train, y_train = load_dataset(os.path.join(training_dir, 'training.csv'), ',')
    x_val, y_val     = load_dataset(os.path.join(validation_dir, 'validation.csv'), ',')
    
    cls = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='auc', 
        max_depth=max_depth
    )
                                    
    cls.fit(x_train, y_train)
    auc = cls.score(x_val, y_val)
    print("AUC ", auc)
    
    cls.save_model(os.path.join(model_dir, 'xgb.model'))
    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
                           
                                    
        
