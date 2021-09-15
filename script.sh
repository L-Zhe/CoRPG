python createVocab.py --file /home/linzhe/document-level-paraphrase/data/pesu/bpe/train.split.bpe \
                             /home/linzhe/document-level-paraphrase/data/pesu/bpe/train.pesu.split.bpe\
                      --save_path ./data/vocab/vocab.share

python preprocess.py --source /home/linzhe/document-level-paraphrase/data/pesu/bpe/train.pesu.comb.bpe \
                     --graph /home/linzhe/document-level-paraphrase/data/pesu/train.pesu.graph \
                     --target /home/linzhe/document-level-paraphrase/data/pesu/bpe/train.comb.bpe \
                     --vocab ./data/vocab/vocab.share \
                     --save_file ./data/para.pair

python preprocess.py --source /home/linzhe/document-level-paraphrase/data/pesu/bpe/test.comb.bpe \
                     --graph /home/linzhe/document-level-paraphrase/data/pesu/test.graph \
                     --vocab ./data/vocab/vocab.share \
                     --save_file ./data/sent.pt


python train.py --cuda_num 0 1 2 3\
                --vocab ./data_roc/vocab/vocab.share\
                --train_file ./data_roc/para.pair\
                --checkpoint_path ./data_roc/model_2100 \
                --restore_file ./data_roc/checkpoint97.pkl \
                --batch_print_info 200 \
                --grad_accum 1 \
                --graph_eps 0.5 \
                --max_tokens 5000

5000/


python generator.py --cuda_num 4 \
                 --file ./data/sent.pt\
                 --ref_file /home/linzhe/document-level-paraphrase/data/eval_data/test.src \
                 --max_tokens 10000 \
                 --vocab ./data/vocab/vocab.share \
                 --decode_method greedy \
                 --beam 5 \
                 --model_path ./data/model/checkpoint29.pkl \
                 --output_path ./data/output \
                 --max_length 300

# de2en

python avg_param.py --input ./data/en2de/model/checkpoint89.pkl \
                            ./data/en2de/model/checkpoint85.pkl \
                            ./data/en2de/model/checkpoint86.pkl \
                            ./data/en2de/model/checkpoint87.pkl \
                            ./data/en2de/model/checkpoint88.pkl \
                    --outputata/en2de/model/checkpoint.pkl

