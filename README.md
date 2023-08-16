# BoundaryDetectionFromHybridText

1.These are the implemented codes and hybrid text dataset for our boundary detection model TriBERT (https://arxiv.org/abs/2307.12267).

2.For details about the boundary detection model TriBERT and how the hybrid essay dataset was constructed.Please refer to our paper:

      **Towards Automatic Boundary Detection for Human-AI Collaborative Hybrid Essay in Education**
      BibTex:
      @article{
          zeng2023towards,
          title={Towards Automatic Boundary Detection for Human-AI Collaborative Hybrid Essay in Education},
          author={Zeng, Zijie and Sha, Lele and Li, Yuheng and Yang, Kaixun and Ga{\v{s}}evi{\'c}, Dragan and Chen, Guanliang},
          journal={arXiv preprint arXiv:2307.12267},
          year={2023}
      }
      
4.Particularly, here we describe the meaning of columns from data.xlsx (hybrid essay dataset)
----------------------------------**********************-----------------------------------

essay_id: The id number of the original source essay.

essayset: the id of the prompt of the source essay.

essay: The original source essay on which the hybrid essay is based.

score1:	The score given by rater 1 for the original source essay.

score2: The score given by rater 2 for the original source essay.

score: The final score for the original source essay.

ratio: Randomly generated number, PLEASE IGNORE THIS.

train_ix: This indicator is used to specify to which set (Train/Valid/Test) the current piece of data belongs.

sent_and_label: The list of <sentence, label> combinations (Each sentence comes from the hybrid essay), i.e., sentences and their labels.  Label here means the 
authorship of the sentence. For example, label 'human' mean human-written and label 'machine' means ChatGPT-generated.

hybrid_text: The human-AI collaboratively written by ChatGPT and students.

boundary_ix: The list containing all boundaries for the above hybrid essay.

boundary_num: The number of boundaries of this hybrid essay.

author_seq: The structure of the hybrid essay. For example, 'H_M' means that the hybrid essay begins with human-written sentences and ends with machine-generated 
sentences (ChatGPT). 'M_H_M' means that the beginning text and ending text are machine-generated while the middle part is human-written.

human_part: Concatenation of all human-written sentences (extracted from the hybrid text).

machine_part: Concatenation of all ChatGPT-generated sentences (extracted from the hybrid text).



----------------------------------**********************-----------------------------------
