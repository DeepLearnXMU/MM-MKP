import os
import csv
import re
from collections import Counter
import time
import argparse
import torch
from my_io import Vocabulary
import numpy as np


def read_src_trg_files(opt, tag="train"):
    '''
    Read data according to the tag (train/valid/test), return a list of (src, trg, img) pairs
    '''
    if tag == "train":
        src_file = opt.train_src
    elif tag == "valid":
        src_file = opt.valid_src
    else:
        src_file = opt.test_src

    tokenized_src = []
    tokenized_trg = []
    img_fns = []

    src_lines = open(src_file, 'r').readlines()

    for line in src_lines:
        # process src and trg line
        src_words = line.strip().split('<sep>')[0].split()
        trg_words = line.strip().split('<sep>')[1].strip().split(';')  # a list of target sequences
        img_fn = line.strip().split('<sep>')[2].strip().strip('\n')

    
        # Append the lines to the data
        tokenized_src.append(src_words)
        tokenized_trg.append(trg_words)
        img_fns.append(img_fn)

    tokenized_pairs = list(zip(tokenized_src, tokenized_trg, img_fns))
    print("Finish reading %d lines of data from %s" % (len(tokenized_src), src_file))
    
    return tokenized_pairs

def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+$'
    text = re.sub(url_re, '', text)
    return text
    
def read_itm(opt,split= 3576,normalize= True):
    
    with open(opt.itm_corpus, encoding='utf-8') as csv_file:
   
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
    
        #csv_reader = csv.DictReader(csv_file)
        tokenized_src = []
        label_list = []
        img_fns = []

        for row in csv_reader:
            text=normalize_text(row['tweet']) if normalize else row['tweet']
            img_fn=f"T{row['tweet_id']}.jpg"
            label=int(row['image_adds'])
            
            # Append the lines to the data
            tokenized_src.append(text)
            img_fns.append(img_fn)
            label_list.append(label)
    
        tokenized_pairs = list(zip(tokenized_src, label_list, img_fns))
        print("Finish reading %d lines of data from %s" % (len(tokenized_src), opt.itm_corpus))

    return tokenized_pairs[:split],tokenized_pairs[split:]

def build_vocab(src_trg_pairs, vocab_size=40000):
    token_cnt = Counter()
    trg_class_cnt = Counter()
    for src_word_list, trg_word_lists, _ in src_trg_pairs:
        token_cnt.update(src_word_list)
        for word_list in trg_word_lists:
            token_cnt.update(word_list.split())
        trg_class_cnt.update(trg_word_lists)

    original_vocab_size = len(token_cnt)
    sorted_word2idx = sorted(token_cnt.items(), key=lambda x: x[1], reverse=True)
    trg_class_sorted_word2idx = sorted(trg_class_cnt.items(), key=lambda x: x[1], reverse=True)

    vocab = Vocabulary()
    trg_class_vocab = Vocabulary()

    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>', '<sep>']
    for word in special_tokens:
        vocab.add_word(word)

    for idx, (word, cnt) in enumerate(sorted_word2idx):
        vocab.add_word(word)
        if idx == vocab_size:
            break
    print('Filtered vocab size: %d / %d' % (vocab_size, original_vocab_size))

    trg_class_vocab.add_word('<unk>')
    for trg_class, cnt in trg_class_sorted_word2idx:
        trg_class_vocab.add_word(trg_class)

    print('Building target class vocab: %d' % len(trg_class_vocab))

    return vocab, trg_class_vocab


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words=15):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    WARNING: if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    Args:
        source_words: list of words (strings)
        word2idx: vocab word2idx
        vocab_size: the maximum acceptable index of word in vocab
    Returns:
        ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
    """
    src_oov = []
    oov_dict = {}
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx['<unk>']
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list


def build_dataset(src_trg_pairs, vocab, trg_class_vocab, is_test):
    examples = []
    for idx, (source, target, img) in enumerate(src_trg_pairs):
        src = [vocab(w) for w in source]
        trgs = [[vocab(w) for w in trg.split()] for trg in target]
        trgs_class = [trg_class_vocab(trg) for trg in target]
        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, vocab.word2idx, len(vocab))
        trg_copies = [[oov_dict[w] if w in oov_dict else vocab(w) for w in trg.split()] for trg in target]

        if is_test:
            examples.append({'src': src, 'trg': trgs[0], 'trg_class': trgs_class[0], 'img': img, 'src_oov': src_oov,
                             'oov_list': oov_list,
                             'trg_copy': trg_copies[0], 'src_str': source, 'trg_str': target})
        else:
            examples.extend(
                [{'src': src, 'trg': trg, 'trg_class': trg_class, 'img': img, 'src_oov': src_oov, 'oov_list': oov_list,
                  'trg_copy': trg_copy, 'src_str': source, 'trg_str': target}
                 for trg, trg_class, trg_copy in zip(trgs, trgs_class, trg_copies)])
    return examples

def build_itm_dataset(src_trg_pairs, vocab):
    examples = []
    for idx, (source, label, img) in enumerate(src_trg_pairs):
        src = [vocab(w) for w in source]
        src_oov, _, oov_list = extend_vocab_OOV(source, vocab.word2idx, len(vocab))
        
        examples.append({'src': src, 'label': label,  'img': img, 'src_oov': src_oov,
                             'oov_list': oov_list,'src_str': source})
    return examples


def main(opt):
    t0 = time.time()
    # Tokenize training data, return a list of tuple
    tokenized_train_pairs = read_src_trg_files(opt, "train")

    # Build vocabulary from training src and trg
    print("Building vocabulary from training data")
    vocab, trg_class_vocab = build_vocab(tokenized_train_pairs, opt.vocab_size)
    #if opt.is_debug
  
    itm_train_pairs,itm_test_pairs = read_itm(opt)
   
    itm_train_data = build_itm_dataset(itm_train_pairs, vocab)
    torch.save(itm_train_data, open(opt.res_data_dir + '/train_itm.pt', 'wb'))

    itm_test_data = build_itm_dataset(itm_test_pairs, vocab)
    torch.save(itm_test_data, open(opt.res_data_dir + '/test_itm.pt', 'wb'))



    print('#pairs of itm train data  = %d' % len(itm_train_data))
    print('#pairs of itm test data  = %d' % len(itm_test_data))

    print('\nFinish and take %.2f seconds' % (time.time() - t0))

    emb_cmd = 'python prepare_tw_glove_emb.py -data_tag %s' % opt.data_tag
    print('\nRunning command: %s' % emb_cmd)
    os.system(emb_cmd)

    return


if __name__ == "__main__":
    data_tag = 'tw_mm_s1_ocr'
    data_suffix='_ocr_kw'

    #data_suffix='_ocr_wiki'
    parser = argparse.ArgumentParser(description='preprocess.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', type=str, default='../data/{}'.format(data_tag), help='The source dir of the data')
    parser.add_argument('-vocab_size', type=int, default=45000, help='The source dir of the data')

    opt = parser.parse_args()
    opt.data_tag = data_tag
    opt.is_debug=True
    #opt.train_src = opt.data_dir + '/train_src.txt'
    #opt.valid_src = opt.data_dir + '/valid_src.txt'
    #opt.test_src = opt.data_dir + '/test_src.txt'

    opt.train_src = opt.data_dir + '/train{}.txt'.format(data_suffix)
    opt.valid_src = opt.data_dir + '/valid{}.txt'.format(data_suffix)
    opt.test_src = opt.data_dir + '/test{}.txt'.format(data_suffix)

    
    opt.itm_corpus='/textimage-data.csv'
    
    processed_data_dir = '../processed_data'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    opt.res_data_dir = "{}/{}".format(processed_data_dir, data_tag)
    if not os.path.exists(opt.res_data_dir):
        os.mkdir(opt.res_data_dir)

    main(opt)

