import model
import data_utils
import inter_utils
import pickle
import time
import os
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from utils import USE_CUDA

from utils import get_torch_variable_from_np, get_data
from scorer import eval_train_batch, pruning_eval_data
from data_utils import output_predict

from data_utils import *

def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)


def make_parser():

    parser = argparse.ArgumentParser(description='A Unified Syntax-aware SRL model')

    # input
    parser.add_argument('--train_data', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--valid_data', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--test_data', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--ood_data', type=str, help='OOD Dataset with CoNLL09 format')

    parser.add_argument('--seed', type=int, default=100, help='the random seed')

    # this default value is from PATH LSTM, you can just follow it too
    # if you want to do the predicate disambiguation task, you can replace the accuracy with yours.
    parser.add_argument('--dev_pred_acc', type=float, default=0.9477,
                            help='Dev predicate disambiguation accuracy')
    parser.add_argument('--test_pred_acc', type=float, default=0.9547,
                            help='Test predicate disambiguation accuracy')
    parser.add_argument('--ood_pred_acc', type=float, default=0.8618,
                            help='OOD predicate disambiguation accuracy')

    # preprocess
    # parser.add_argument('--preprocess', action='store_true',
    #                     help='Preprocess')
    parser.add_argument('--tmp_path', type=str, help='temporal path')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--result_path', type=str, help='result path')
    parser.add_argument('--pretrain_embedding', type=str, help='Pretrain embedding like GloVe or word2vec')
    parser.add_argument('--pretrain_emb_size', type=int, default=100,
                        help='size of pretrain word embeddings')

    # train 
    parser.add_argument('--train', action='store_true',
                            help='Train')
    parser.add_argument('--epochs', type=int, default=20,
                            help='Train epochs')
    parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout when training')
    parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size in train and eval')
    parser.add_argument('--word_emb_size', type=int, default=100,
                            help='Word embedding size')
    parser.add_argument('--pos_emb_size', type=int, default=32,
                            help='POS tag embedding size')
    parser.add_argument('--lemma_emb_size', type=int, default=100,
                            help='Lemma embedding size')
    parser.add_argument('--deprel_emb_size', type=int, default=64,
                            help='Dependency relation embedding size')
    parser.add_argument('--bilstm_hidden_size', type=int, default=512,
                            help='Bi-LSTM hidden state size')
    parser.add_argument('--bilstm_num_layers', type=int, default=4,
                            help='Bi-LSTM layer number')
    parser.add_argument('--valid_step', type=int, default=1000,
                            help='Valid step size')

    parser.add_argument('--use_highway', action='store_true',
                        help='[USE] highway connection')
    parser.add_argument('--highway_num_layers',type=int, default=10,
                            help='Highway layer number')

    parser.add_argument('--use_self_attn', action='store_true',
                        help='[USE] self attention')
    parser.add_argument('--self_attn_heads',type=int, default=10,
                            help='Self attention Heads')

    parser.add_argument('--use_flag_emb', action='store_true',
                        help='[USE] predicate flag embedding')
    parser.add_argument('--flag_emb_size',type=int, default=16,
                            help='Predicate flag embedding size')

    parser.add_argument('--use_elmo_emb', action='store_true',
                        help='[USE] ELMo embedding')
    parser.add_argument('--elmo_emb_size',type=int, default=300,
                            help='ELMo embedding size')
    parser.add_argument('--elmo_options',type=str,
                            help='ELMo options file')
    parser.add_argument('--elmo_weight',type=str,
                            help='ELMo weight file')

    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')

    # syntactic
    
    parser.add_argument('--use_gcn', action='store_true',
                        help='[USE] GCN')
    parser.add_argument('--use_sa_lstm', action='store_true',
                        help='[USE] Syntax-aware LSTM')
    parser.add_argument('--use_tree_lstm', action='store_true',
                        help='[USE] tree LSTM')
    parser.add_argument('--use_rcnn', action='store_true',
                        help='[USE] RCNN')

    # eval
    parser.add_argument('--eval', action='store_true',
                            help='Eval')
    parser.add_argument('--model', type=str, help='Model')

    return parser

if __name__ == '__main__':
    print('Unified Syntax-aware SRL model')

    args = make_parser().parse_args()

    # set random seed
    seed_everything(args.seed, USE_CUDA)

    # do preprocessing
    # if args.preprocess:
    train_file = args.train_data
    dev_file = args.valid_data
    test_file = args.test_data
    test_ood_file = args.ood_data

    tmp_path = args.tmp_path

    if tmp_path is None:
        print('Fatal error: tmp_path cannot be None!')
        exit()

    print('start preprocessing data...')

    start_t = time.time()

    # make word/pos/lemma/deprel/argument vocab
    print('\n-- making (word/lemma/pos/argument/predicate) vocab --')
    vocab_path = tmp_path
    print('word:')
    make_word_vocab(train_file,vocab_path, unify_pred=unify_pred)
    print('pos:')
    make_pos_vocab(train_file,vocab_path, unify_pred=unify_pred)
    print('lemma:')
    make_lemma_vocab(train_file,vocab_path, unify_pred=unify_pred)
    print('deprel:')
    make_deprel_vocab(train_file,vocab_path, unify_pred=unify_pred)
    print('argument:')
    make_argument_vocab(train_file, dev_file, test_file, vocab_path, unify_pred=unify_pred)
    print('predicate:')
    make_pred_vocab(train_file, dev_file, test_file, vocab_path)

    deprel_vocab = load_deprel_vocab(os.path.join(tmp_path, 'deprel.vocab'))

    # shrink pretrained embeding
    print('\n-- shrink pretrained embeding --')
    pretrain_file = args.pretrain_embedding
    pretrained_emb_size = args.pretrain_emb_size
    pretrain_path = tmp_path
    shrink_pretrained_embedding(train_file, dev_file, test_file, pretrain_file, pretrained_emb_size, pretrain_path)

    train_res = make_dataset_input(train_file, os.path.join(tmp_path,'train.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab)
    dev_res = make_dataset_input(dev_file, os.path.join(tmp_path,'dev.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab)
    test_res = make_dataset_input(test_file, os.path.join(tmp_path,'test.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab)
    if test_ood_file is not None:
        ood_res = make_dataset_input(test_ood_file, os.path.join(tmp_path,'test_ood.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab)

    print('\t data preprocessing finished! consuming {} s'.format(int(time.time()-start_t)))

    print('\t start loading data...')
    start_t = time.time()

    train_input_file = os.path.join(os.path.dirname(__file__),'temp/train.input')
    dev_input_file = os.path.join(os.path.dirname(__file__),'temp/dev.input')
    test_input_file = os.path.join(os.path.dirname(__file__),'temp/test.input')
    if test_ood_file is not None:
        test_ood_input_file = os.path.join(os.path.dirname(__file__),'temp/test_ood.input')
    
    train_dataset = data_utils.load_dataset_input(train_input_file)
    dev_dataset = data_utils.load_dataset_input(dev_input_file)
    test_dataset = data_utils.load_dataset_input(test_input_file)
    if test_ood_file is not None:
        test_ood_dataset = data_utils.load_dataset_input(test_ood_input_file)

    word2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/word2idx.bin'))
    idx2word = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2word.bin'))

    lemma2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/lemma2idx.bin'))
    idx2lemma = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2lemma.bin'))

    pos2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pos2idx.bin'))
    idx2pos = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2pos.bin'))

    deprel2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/deprel2idx.bin'))
    idx2deprel = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2deprel.bin'))

    pretrain2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pretrain2idx.bin'))
    idx2pretrain = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2pretrain.bin'))

    argument2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/argument2idx.bin'))
    idx2argument = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2argument.bin'))

    pretrain_emb_weight = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pretrain.emb.bin'))

    print('\t data loading finished! consuming {} s'.format(int(time.time()-start_t)))

    #result_path = os.path.join(os.path.dirname(__file__),'result/')

    print('\t start building model...')
    start_t = time.time()

    dev_predicate_sum = dev_res[0]
    test_predicate_sum = test_res[0]

    if test_ood_file is not None:
        test_ood_predicate_sum = ood_res[0]

    dev_predicate_correct = int(dev_predicate_sum * args.dev_pred_acc)
    test_predicate_correct = int(test_predicate_sum * args.test_pred_acc)

    if test_ood_file is not None:
        test_ood_predicate_correct = int(test_ood_predicate_sum * args.ood_pred_acc)

    # hyper parameters
    max_epoch = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    dropout = args.dropout
    word_embedding_size = args.word_emb_size
    pos_embedding_size = args.pos_emb_size
    pretrained_embedding_size = args.pretrain_emb_size
    lemma_embedding_size = args.lemma_emb_size

    use_deprel = args.use_deprel
    deprel_embedding_size = args.deprel_emb_size
    

    bilstm_hidden_size = args.bilstm_hidden_size
    bilstm_num_layers = args.bilstm_num_layers
    show_steps = args.valid_step
    
    use_highway = args.use_highway
    highway_layers = args.highway_num_layers


    use_flag_embedding = args.use_flag_emb
    flag_embedding_size = args.flag_emb_size


    use_elmo = args.use_elmo
    elmo_embedding_size = args.elmo_emb_size
    elmo_options_file = args.elmo_options_file
    elmo_weight_file = args.elmo_weight_file


    use_self_attn = args.use_self_attn
    self_attn_head = args.self_attn_heads
    

    use_tree_lstm = args.use_tree_lstm
    use_sa_lstm = args.use_sa_lstm
    use_gcn = args.use_gcn
    use_rcnn = args.use_rcnn

    if args.train:
        FLAG = 'TRAIN'
    if args.Valid:
        FLAG = 'EVAL'
        MODEL_PATH = args.model

    if FLAG == 'TRAIN':
        model_params = {
            "dropout":dropout,
            "batch_size":batch_size,
            "word_vocab_size":len(word2idx),
            "lemma_vocab_size":len(lemma2idx),
            "pos_vocab_size":len(pos2idx),
            "deprel_vocab_size":len(deprel2idx),
            "pretrain_vocab_size":len(pretrain2idx),
            "word_emb_size":word_embedding_size,
            "lemma_emb_size":lemma_embedding_size,
            "pos_emb_size":pos_embedding_size,
            "pretrain_emb_size":pretrained_embedding_size,
            "pretrain_emb_weight":pretrain_emb_weight,
            "bilstm_num_layers":bilstm_num_layers,
            "bilstm_hidden_size":bilstm_hidden_size,
            "target_vocab_size":len(argument2idx),
            "use_highway":use_highway,
            "highway_layers": highway_layers,
            "use_self_attn":use_self_attn,
            "self_attn_head":self_attn_head,
            "use_deprel":use_deprel,
            "deprel_emb_size":deprel_embedding_size,
            "deprel2idx":deprel2idx,
            "use_flag_embedding":use_flag_embedding,
            "flag_embedding_size":flag_embedding_size,
            'use_elmo':use_elmo,
            "elmo_embedding_size":elmo_embedding_size,
            "elmo_options_file":elmo_options_file,
            "elmo_weight_file":elmo_weight_file,
            "use_tree_lstm":use_tree_lstm,
            "use_gcn":use_gcn,
            "use_sa_lstm":use_sa_lstm,
            "use_rcnn":use_rcnn   
        }

        # build model
        srl_model = model.End2EndModel(model_params)

        if USE_CUDA:
            srl_model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(srl_model.parameters(),lr=learning_rate)

        print(srl_model)

        print('\t model build finished! consuming {} s'.format(int(time.time()-start_t)))

        print('\nStart training...')

        dev_best_score = None
        test_best_score = None
        test_ood_best_score = None

        for epoch in range(max_epoch):

            epoch_start = time.time()
            for batch_i, train_input_data in enumerate(inter_utils.get_batch(train_dataset, batch_size,word2idx,
                                                                    lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, shuffle=True)):
                

                target_argument = train_input_data['argument']
                
                flat_argument = train_input_data['flat_argument']

                target_batch_variable = get_torch_variable_from_np(flat_argument)

                bs = train_input_data['batch_size']
                sl = train_input_data['seq_len']
                
                out = srl_model(train_input_data)

                loss = criterion(out, target_batch_variable)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_i > 0 and batch_i % show_steps == 0: 

                    _, pred = torch.max(out, 1)

                    pred = get_data(pred)

                    # pred = pred.reshape([bs, sl])

                    print('\n')
                    print('*'*80)

                    eval_train_batch(epoch, batch_i, loss.data[0], flat_argument, pred, argument2idx)

                    print('dev:')
                    score, dev_output = eval_data(srl_model, criterion, dev_dataset, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred, dev_predicate_correct, dev_predicate_sum)
                    if dev_best_score is None or score[2] > dev_best_score[2]:
                        dev_best_score = score
                        output_predict(os.path.join(result_path,'dev_argument_{:.2f}.pred'.format(dev_best_score[2]*100)),dev_output)
                        # torch.save(srl_model, os.path.join(os.path.dirname(__file__),'model/best_{:.2f}.pkl'.format(dev_best_score[2]*100)))
                    print('\tdev best P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(dev_best_score[0] * 100, dev_best_score[1] * 100,
                                                                                                    dev_best_score[2] * 100, dev_best_score[3] * 100,
                                                                                                    dev_best_score[4] * 100, dev_best_score[5] * 100))

                
                print('\repoch {} batch {} batch consume:{} s'.format(epoch, batch_i, int(time.time()-epoch_start)), end="")
                epoch_start = time.time()

    else:
        srl_model = torch.load(MODEL_PATH)
        srl_model.eval()
        print('test:')
        score, test_output = eval_data(srl_model, criterion, test_dataset, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred, test_predicate_correct, test_predicate_sum)
        print('\ttest best P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(test_best_score[0] * 100, test_best_score[1] * 100,
                                                                                        test_best_score[2] * 100, test_best_score[3] * 100,
                                                                                        test_best_score[4] * 100, test_best_score[5] * 100))

        if test_ood_file is not None: 
            print('ood:')
            score, ood_output = eval_data(srl_model, criterion, test_ood_dataset, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred, test_ood_predicate_correct, test_ood_predicate_sum)
            output_predict(os.path.join(result_path,'ood_argument_{:.2f}.pred'.format(score[2]*100)),ood_output)
            print('\tood P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(score[0] * 100, score[1] * 100,
                                                                                            score[2] * 100, score[3] * 100,
                                                                                            score[4] * 100, score[5] * 100))

