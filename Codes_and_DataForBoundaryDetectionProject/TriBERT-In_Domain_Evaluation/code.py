
from nltk.tokenize import sent_tokenize
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import ml_metrics as metrics
import datetime
import numpy as np
import torch
import time, os
import random
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import random
from scipy.spatial import distance
import argparse
import ast


def get_sentence(text):
    """
    from nltk.tokenize import sent_tokenize
    import nltk
    """
    res = sent_tokenize(text)
    return [s.replace('\n',' ').strip() for s in res]



def get_euclidean_distance(v1,v2):
    return round(distance.euclidean(v1,v2),4)



def get_one_triplet_example(data, target_essayset):
    
    try: 
        essayset = random.choice( 
                [e for e in [1,2,3,4,5,6,7,8] if e!= 99999 ]
        )

        random_record = data[(data.essayset == essayset) & (data.train_ix == 'train')].sample(n = 1).to_dict(orient='records')[0] 

        pos = get_sentence(random_record['human_part'])
        neg = get_sentence(random_record['machine_part'])
        
        if random.random() > 0.5:#0.5:
            p_sents = random.sample(pos,1)
            #random.shuffle(p_sents)
            n_sents = random.sample(neg, 2)
            random.shuffle(n_sents)
            example = InputExample(texts=[n_sents[0], n_sents[1], p_sents[0]])
        else:
            # there might be bugs due to the length of p_sent being 1 (but you try to pick 2 elements from it)
            # in this case, we just try again until p_sent has its length > 1
            p_sents = random.sample(pos,2)
            random.shuffle(p_sents)
            n_sents = random.sample(neg, 1)
            #random.shuffle(n_sents)
            example = InputExample(texts=[p_sents[0], p_sents[1], n_sents[0]])
        return example
    
    except:
        #print('error but fixed')
        return get_one_triplet_example(data, target_essayset)

    
def get_one_epoch_train_examples(data, target_essayset,examples_per_epoch):
    res=[]
    for i in range(examples_per_epoch):
        res.append( get_one_triplet_example(data, target_essayset) )
        
    return res




   
    
def predict_boundary_for_one_essay(model, essay):
    sents = get_sentence(essay)

    distance_result=[]
    
    sents_emb = []
    for sent in sents:
        emb = model.encode([sent]).flatten()
        sents_emb.append(emb)
        
    #prototype_size = 3

    for i in range(len(sents_emb)-1):
        emb1 = sents_emb[max(i+1-prototype_size, 0): i+1]
        emb2 = sents_emb[i+1: i+1+prototype_size]
        
        p1 = np.mean(emb1, axis=0)
        p2 = np.mean(emb2, axis=0)
        dist = get_euclidean_distance(p1, p2)
        distance_result.append(dist)

    #boundary= distance_result.index(max(distance_result)) + 1 # this is the predicted boundary_ix, meaning that (boundary_ix)th is the last human sentence.
    
    boundary = sorted([(dist, ix+1) for ix, dist in enumerate(distance_result)], key=lambda x:x[0], reverse = True)
    boundary = [ix for dist,ix in boundary][:topk]
    
    if len(boundary) < topk:
       boundary = boundary + random.sample([ i+1 for i in range(len(distance_result)) if (i+1) not in boundary] , topk - len(boundary))
       boundary = sorted(boundary)
        
    
    random_boundary = random.sample( [ i+1 for i in range(len(distance_result))],topk )
    
    #random_boundary = random.sample( list(range(len(sent_predictions)))[1:] ,topk )
    
    return boundary, random_boundary


def get_evaluation_result(predicted,labels):

    
    tpl = list(zip(predicted,labels))
    result={'precision':[],'recall':[],'f1_score':[]}
    
    for p, l in tpl:
        intersection = set(p) & set(l)
        precision = 0.0001+len(intersection) * 1.0 / topk
        recall = 0.0001+len(intersection) * 1.0 / len(l)
       
        result['precision'].append(  precision ) 
        result['recall'].append( recall ) 
        result['f1_score'].append(  2 * precision * recall*1.0 / (precision + recall) )   #f1_score = 2 * precision * recall*1.0 / (precision + recall)
    
    return round( np.mean(result['precision']) ,3), round( np.mean(result['recall']) ,3), round( np.mean(result['f1_score']) ,3)


def evaluate_model(model, data, targetprompt, test_type):
    print('--------Start Evaluating: {}--------'.format(str(datetime.datetime.now())[:19]))
    if test_type == 'valid':
        prompt_data = data[
            #(data.essayset != targetprompt)
            #&
            (data.train_ix == 'valid')
            #&
            #(data.ratio > 0.81)
        ]
    elif test_type == 'test':  # that is 'test' type on unseen prompt
        prompt_data = data[
            (data.train_ix == 'test')
        ]
    
    all_hybird_essay = list( prompt_data['hybrid_text'])
    #all_labels = list(prompt_data['boundary_ix'])
    all_labels = [ast.literal_eval(e) for e in list(prompt_data['boundary_ix'])] # each elements is a list
    all_predicted_boundary = []
    random_guess=[]
    
    for essay in all_hybird_essay:
        
        predicted_boundary,rn_boundary = predict_boundary_for_one_essay(model, essay)
        all_predicted_boundary.append(predicted_boundary)
        random_guess.append(rn_boundary)
        
    precision, recall, f1_score = get_evaluation_result(all_predicted_boundary, all_labels)
    rn_precision, rn_recall, rn_f1_score = get_evaluation_result(random_guess, all_labels)
        
    print('--------------------{}ing------------------------'.format(test_type))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('f1_score: {}'.format(f1_score))
    print('-----Random-------')
    print('rn_Precision: {}'.format(rn_precision))
    print('rn_Recall: {}'.format(rn_recall))
    print('rn_f1_score: {}'.format(rn_f1_score))
        
    if test_type == 'test':
        result_file = 'lr_{}_predicted_results_{}.xlsx'.format(lr, targetprompt)
        prompt_data.to_excel(result_file, index=None)
        prompt_data = pd.read_excel(result_file)
        prompt_data['predicted_boundary'] = pd.Series(all_predicted_boundary)
        prompt_data['random_boundary'] = pd.Series(random_guess)
        prompt_data.to_excel(result_file, index=None)
        
    print('--------Ending Evaluating: {}--------'.format(str(datetime.datetime.now())[:19]))
        
    return f1_score
        
    



def train_model(model, data, lr, num_epochs, examples_per_epoch, current_prompt,save_model = 1):
    
    best_metric_so_far = None
    best_model = 'lr{}_bd_{}.bert'.format(lr,current_prompt)

    for i in range(num_epochs):
        print('---------Epoch {} ----time: {}'.format(i + 1,str(datetime.datetime.now())[:19]))

        train_examples = get_one_epoch_train_examples(data, current_prompt, examples_per_epoch)
        train_dataloader = DataLoader(train_examples, shuffle = True, batch_size = 1)
        train_loss = losses.TripletLoss(model = model,triplet_margin = 1) # euclidean distance

        model.fit(
            train_objectives = [(train_dataloader, train_loss)],
            #evaluator = evaluator,
            epochs = 1,
            scheduler = 'Constantlr',
            weight_decay = 0.01,
            show_progress_bar = False,#True,
            warmup_steps = 0,
            optimizer_params = {'lr': lr}
        )
        lr = lr * 0.8

        print('--------now evaluating on valid set----------')
        f1 = evaluate_model(model, data, current_prompt, test_type = 'valid')#evaluate_model

        if best_metric_so_far is None or best_metric_so_far < f1:

            old_f1 = best_metric_so_far
            best_metric_so_far = f1

            if os.path.exists(best_model):
                os.remove(best_model)
                print('---old model removed---')

            print('---Best f1 has been updated from {} to {}---, new best model saved'.format(round(old_f1, 4) if old_f1 is not None else None, round(best_metric_so_far,4)))
            torch.save(model,best_model)

        else:
            print('---Old f1 {} > New f1 {}---'.format(round(best_metric_so_far, 4), round(f1, 4)))
            print('No improvement detected compared to last validation round, early stop is triggered.')
            model = torch.load(best_model)
            break

    print('--------now testing on unseen prompt {}----------'.format(current_prompt))
    evaluate_model(model, data, current_prompt, test_type = 'test')

    if os.path.exists(best_model) and save_model !=1:
        os.remove(best_model)
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="This is a description")
    parser.add_argument('--num_epochs',dest='num_epochs',required = True, type = int)
    parser.add_argument('--prototype_size',dest='prototype_size',required = False, type = int)
    parser.add_argument('--topk',dest='topk',required = False, type = int)
    parser.add_argument('--save_model',dest='save_model',required = False, type = int)
    parser.add_argument('--learning_rate',dest='learning_rate',required = True, type = float) 
    parser.add_argument('--data_file', dest='data_file', required = True, type = str)  
    parser.add_argument('--targetprompt', dest='targetprompt', required = False, type = int)
    parser.add_argument('--examples_per_epoch',dest='examples_per_epoch',required = True, type = int)

    args = parser.parse_args()  

    
    num_epochs = args.num_epochs
    prototype_size = args.prototype_size if args.prototype_size is not None else 1
    topk = args.topk if args.topk is not None else 3
    save_model = args.save_model if args.save_model is not None else 0
    lr = args.learning_rate
    data_file = args.data_file
    current_prompt = args.targetprompt
    examples_per_epoch = args.examples_per_epoch
    
    
    print('num_epochs: {}'.format(num_epochs))
    print('prototype_size: {}'.format(prototype_size))
    print('topk: {}'.format(topk))
    print('save_model: {}'.format(True if save_model == 1 else False))
    print('learning_rate: {}'.format(lr))
    print('data_file: {}'.format(data_file))
    print('target_prompt: {}'.format(current_prompt))
    print('examples_per_epoch: {}'.format(examples_per_epoch))
  

    data = pd.read_excel(data_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2',device = device)#, device='cpu' )

    
    train_model(
            model = model, 
            data = data, 
            lr = lr, 
            num_epochs = num_epochs, 
            examples_per_epoch = examples_per_epoch, 
            current_prompt = current_prompt,
            save_model = save_model
    )
    
    