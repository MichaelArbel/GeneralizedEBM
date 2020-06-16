





# Training the model 
python main.py --config=configs/training.yaml --dataset=cifar10


# Sampling from trained model
python main.py --config=configs/sampling.yaml --dataset=cifar10 --latent_sampler=langevin --lmc_gamma=0.0001


