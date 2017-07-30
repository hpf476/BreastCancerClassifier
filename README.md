# BreastCancerClassifier
Code for Ai Project 50.021 on detecting malignant breast tumor cells


AMI ID: 50.021project

Instance type: p2-xlarge

Command to activate tensorflow: source ~/workspace_ami/tensorflow.env/bin/activate

Neccessary step: copy the ./tmp folder into ./finetune directory

Command for redirecting console output into a txt file: python script.py &> yourfile.txt




Command for running inception code:

python retrain.py --image_dir=.... --list_dir=... --zoom_level=40X --intermediate_store_frequency=500 --how_many_training_steps=4000 --learning_rate=0.01 --train_batch_size=100 --flip_left_right=False --random_crop=0 --random_scale=0 --random_brightness=0
