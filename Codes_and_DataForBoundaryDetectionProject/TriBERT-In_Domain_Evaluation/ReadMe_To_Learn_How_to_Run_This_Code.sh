
#--topk: the length of the candidate list of the possible boundary indexes. 
#--Num_epochs: The number of epochs for encoder training
#--Examples_per_epoch: Number of examples for each epoch
#--Learning_rate: Learning rate
#--Data_file data: Data source file (That's to say, the formatted hybrid essay dataset)
#--Save_model: Whether to save model or not. '0' means NO; '1' means YES
#--Prototype_size: The number of consecutive sentences needed to calculate the averaged embedding (i.e., prototype).
#


# Run the following script to training our model and then perform boundary detection
python -u code.py --num_epochs 10 --learning_rate 0.00001 --data_file data.xlsx  --examples_per_epoch 5000 --save_model 0  --prototype_size 2 > results.txt




