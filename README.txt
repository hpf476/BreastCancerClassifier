This is a git repo for AI project on Malign Breast Cancer Detection

AMI ID: 50.021project

Instance type: p2-xlarge

Command to activate tensorflow: source ~/workspace_ami/tensorflow.env/bin/activate

Neccessary step: copy the ./tmp folder into ./finetune directory

Command for redirecting console output into a txt file: python script.py &> yourfile.txt




Command for running inception code:

python retrain_inception.py --image_dir='/home/ubuntu/workspace_ami/breakhis_data/' --list_dir='/home/ubuntu/workspace_ami/breakhis_data/train_val_test_60_12_28/non_shuffled/split1/' --intermediate_store_frequency=500 --learning_rate=0.01 --flip_left_right=False --random_crop=0 --random_scale=0 --random_brightness=0



To review GPU usage: nvidia-smi

To kill a process: kill -9 PID