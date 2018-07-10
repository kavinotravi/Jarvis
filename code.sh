THEANO_FLAGS=device=gpu,floatX=float32 python train.py ../train_data.json ../valid_data.json --epochs 3 ../final_model --continue_training True
