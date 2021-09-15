# CoRPG
Code for paper Document-Level Paraphrase Generation with Sentence Rewriting and Reordering by Zhe Lin, Yitao Cai and Xiaojun Wan. This paper is accepted by Findings of EMNLP'21.

<img src="https://github.com/L-Zhe/CoPRG/blob/main/img/model.jpg?raw=true" width = "800" alt="overview" align=center />

## Datasets

We leverage [ParaNMT](https://www.cs.cmu.edu/~jwieting/) to train a sentence-level paraphrasing model. We select [News-Commentary](http://www.statmt.org/wmt20/ translation- task.html) as document corpora, and we employ sentence-level paraphrasing model to generate a pseudo document-level paraphrase and use ALBERT to generate its coherence relationship graph. All these data are released at [here](https://github.com/L-Zhe/CoPRG/data).

## System Output

If you are looking for system output and don't bother to install dependencies and train a model (or run a pre-trained model), the [all-system-output](https://github.com/L-Zhe/CoPRG/all-system-output) folder is for you.

## Dependencies

>PyTorch >= 1.4
>
>Transformers == 4.1.1
>
>nltk == 3.5
>
>tqdm 
>
>torch_optimizer == 0.1.0

## Train a Document-Level Paraphrase Model

### Step1: Prepare dataset

We release the dataset we used in [data folder](https://github.com/L-Zhe/CoPRG/data). If you want to use your own dataset, you need to follow the following procedure.

First, you should train a sentence-level paraphrase model to generate pseudo documen paraphrase dataset (We leverage paraNMT and [fairseq]() to train this model).

Then,  you should download the [ALBERT](https://huggingface.co/albert-base-v2) model and fine-tuning it with your own dataset with the following script:

```shell
python eval/coherence.py --train
												 --pretrain_model	[pretrain_model file]
												 --save_file [the path to save fine-tune model]
												 --text_file [the corpora used to fine-tune the pretrain_model] 
```

We also provide our fine-tune model in [here]().

Finally, you can leveraged ALBERT to generate the coherence relationship graph:

```shell
python eval/coherence.py --inference
												 --pretrain_model	[pretrain_model file]
												 --text_file [generate the coherence relationship graph of this corpora]
```

**NOTE：**Our code only supports to generate the paraphrasing of documents with 5 sentences. If you want to generate longer or variable length document paraphrase, you need to make some modifications to the code.

### Step2: Process dataset

Create Vocabulary:

```shell
python createVocab.py --file ./data/news-commentary/data/bpe/train.split.bpe \
                             ./data/news-commentary/data/bpe/train.pesu.split.bpe\
                      --save_path ./data/vocab.share
```

Processing Training Dataset:

```shell
python preprocess.py --source ./data/news-commentary/data/bpe/train.pesu.comb.bpe \
                     --graph ./data/news-commentary/data/train.pesu.graph \
                     --target ./data/news-commentary/data/bpe/train.comb.bpe \
                     --vocab ./data/vocab.share \
                     --save_file ./data/para.pair
```

Processing Test Dataset:

```shell
python preprocess.py --source ./data/news-commentary/data/bpe/test.comb.bpe \
                     --graph ./data/news-commentary/data/test.graph \
                     --vocab ./data/vocab.share \
                     --save_file ./data/sent.pt
```

### Step3: Train a document-level paraphrase model

```shell
python train.py --cuda_num 0 1 2 3\
                --vocab ./data/vocab.share\
                --train_file ./data/para.pair\
                --checkpoint_path ./data/model \
                --batch_print_info 200 \
                --grad_accum 1 \
                --graph_eps 0.5 \
                --max_tokens 5000
```

### Step4: Generate document-level paraphrase

```shell
python generator.py --cuda_num 4 \
                 		--file ./data/sent.pt\
                 		--ref_file ./data/news-commentary/data/test.comb \
                 		--max_tokens 10000 \
                 		--vocab ./data/vocab.share \
                 		--decode_method greedy \
                 		--beam 5 \
                 		--model_path ./data/model.pkl \
                 		--output_path ./data/output \
                 		--max_length 300
```

## Pre-trained Models

## Evaluation Matrics

We evaluate our model in three aspects: relevancy, diversity, coherence.

### Relevancy

We leverage [BERTScore](https://github.com/Tiiiger/bert_score) to evaluate the semantic relevancy between paraphrase and original sentence.  

### Diversity

We employ self-[TER](https://github. com/jhclark/multeval.) and self-[WER](https://github. com/belambert/asr-evaluation) to evaluate the diversity of our model. 

### Coherence

We raise COH and COH-p to evaluate the coherence of paraphrase as follows:

<img src="https://github.com/L-Zhe/CoPRG/blob/main/img/coh.jpg?raw=true" width = "800" alt="overview" align=center />

where $$P_{SOP}$$ is calculated by [ALBERT](https://openreview.net/forum?id=H1eA7AEtvS). We provide the script for these two evaluation matrics as follow:

```shell
python eval/coherence.py --coh
												 --pretrain_model [the pretrain albert file]
												 --text_file [the corpora to be evaluated]
```

## Results

<img src="https://github.com/L-Zhe/CoPRG/blob/main/img/result.jpg?raw=true" width = "800" alt="overview" align=center />

## Citation

If you use any content of this repo for your work, please cite the following bib entry:

```
@inproceedings{lin-wan-2021-document,
    title = "Document-Level Paraphrase Generation with Sentence Rewriting and Reordering",
    author = "Lin, Zhe, Cai, Yitao  and
      Wan, Xiaojun",
    booktitle = "Findings of EMNLP 2021",
}
```
