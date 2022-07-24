for i in 1 2 3 4 5 6 7 8 9 10
do
    python3 train.py --data_name FashionMNIST --data_root data --hidden_channels 32 --epochs 100 --eval_interval 1 --savedir ensembles --latent_dims 90 --seed $i
done
