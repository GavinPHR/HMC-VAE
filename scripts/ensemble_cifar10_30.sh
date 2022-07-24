for i in 1 2 3 4 5 6 7 8 9 10
do
    python3 train.py --data_name CIFAR10 --data_root data --hidden_channels 64 --epochs 100 --eval_interval 1 --savedir ensembles --latent_dims 30 30 30 --seed $i
done
