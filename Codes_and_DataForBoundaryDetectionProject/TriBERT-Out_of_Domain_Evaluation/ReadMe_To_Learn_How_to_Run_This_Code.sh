

#--topk: the length of the candidate list of the possible boundary indexes. 
#--num_epochs: The number of epochs for encoder training
#--examples_per_epoch: The number of examples for each epoch
#--learning_rate: Learning rate
#--data_file data: Data source file (That's to say, the formatted hybrid essay dataset)
#--save_model: Whether to save model or not. '0' means NO; '1' means YES
#--prototype_size: The number of consecutive sentences needed to calculate the averaged embedding (i.e., prototype).

#--targetprompt: The target prompt to be tested. In our hybrid dataset, for example, when target prompt is 7, it means than the encoder will be trained on prompt 1,2,3,4,5,6,8 (all prompts except 7). However, the model is required to predict on prompt 7. ----> This is the Out-of-Domain evaluation setting.


# Run the following script to training our model and then perform boundary detection
python -u code.py --num_epochs 10 --learning_rate 0.000001 --data_file data.xlsx --targetprompt 1 --examples_per_epoch 5000 --save_model 0  --prototype_size 2 > result.txt






