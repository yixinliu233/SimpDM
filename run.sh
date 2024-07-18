python3 main.py --dataset yacht --epochs 30000 --lr 0.0001 --num_layers 5 --hidden_units 256 --gammas 1.0_0.8_0.001 --ssl_loss_weight 10
python3 main.py --dataset yeast --epochs 10000 --lr 0.0001 --num_layers 3 --hidden_units 256 --gammas 0.5_0.4_0.0001 --ssl_loss_weight 3 
python3 main.py --dataset wine_white --epochs 30000 --lr 0.0001 --num_layers 3 --hidden_units 1024 --gammas 1.0_0.8_0.001 --ssl_loss_weight 0.2
python3 main.py --dataset power --epochs 30000 --lr 0.0001 --num_layers 5 --hidden_units 1024 --gammas 2.0_1.5_0.001 --ssl_loss_weight 1