import torch
import numpy as np
import pickle
import os


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class MultimodalTweetDataset(torch.utils.data.Dataset):

    def __init__(self, examples, vocab, trg_class_vocab, use_text, use_img, use_attribute,use_type=None, use_bert_src=None,
                 img_feats_fn=None, attribute_feats_fn=None, url_map_fn=None,
                 bert_feats_fn=None, src_str_map_fn=None,
                 is_test=False, only_classifier=False, debug=False):
        if debug:
            self.examples = examples[:100]
            print('Load 100 examples for debug mode')
        else:
            self.examples = examples

        # for test the D8_NiX3XoAEHa8s.jpg
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D8_NiX3XoAEHa8s.jpg']
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D7Q6xTPW0AENo16.jpg']
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D3Gk5-eXkAA-7KH.jpg']
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D3BLr7tWoAYiszm.jpg']
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D0NQJv6UwAAWb7m.jpg']
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D5wGsmtW0AAuKEm.jpg']
        # self.examples = [e for e in examples if e['img'].split('/')[-1] == 'D0wn9TnXQAAGWyD.jpg']
        # assert len(self.examples) == 1

        self.vocab = vocab
        self.pad_idx = vocab('<pad>')
        self.trg_class_vocab = trg_class_vocab
        self.trg_class_vocab_size = len(trg_class_vocab)

        self.is_test = is_test
        self.only_classifier = only_classifier

        self.use_text = use_text
        self.use_img = use_img
        self.use_attribute = use_attribute
        self.use_bert_src = use_bert_src
        self.use_type=use_type

        self.num_roi_boxes=36
        self.img_feat_dim=2048
        self.imagefeat_dir="/home/sata/dyfff/CMKP_feature36"

        if use_img or use_attribute:
            with open(url_map_fn, 'rb') as f:
                self.url_map = pickle.load(f)

        if use_img:
            with open(img_feats_fn, 'rb') as f:
                self.img_feats = pickle.load(f)

        if use_attribute:
            with open(attribute_feats_fn, 'rb') as f:
                self.attribute_feats = pickle.load(f)

        if self.use_bert_src:
            with open(src_str_map_fn, 'rb') as f:
                self.src_str_map = pickle.load(f)

            with open(bert_feats_fn, 'rb') as f:
                self.bert_feats = pickle.load(f)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list,type_list=None):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
    
        padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))
         
        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]
        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.pad_idx)
        input_mask = input_mask.type(torch.FloatTensor)

        if type_list is not None:
            assert len(input_list)==len(type_list)
            padded_type=self.pad_idx * np.ones((len(input_list), max_seq_len))
            for k in range(len(input_list)):
                current_len = input_list_lens[k]
                padded_type[k][:current_len] = type_list[k]
            padded_type = torch.LongTensor(padded_type)
            return padded_batch, input_list_lens, input_mask,padded_type

        return padded_batch, input_list_lens, input_mask
    
    def read_npz(self,imagefeat_dir,img_id):

        #print(imagefeat_dir)
        #print(img_id)
        tmp=img_id.split('.')[0]
        #print("img_id",tmp)
        feat_dict=np.load(os.path.join(imagefeat_dir,str(tmp)+'.npz'))
        
        img_feat=feat_dict['x']   #[100,2048]
        img_feat_sum=np.sqrt((img_feat**2).sum())+1e-6
        img_feat=(img_feat/img_feat_sum)
        #img_feat=(img_feat/np.sqrt((img_feat**2).sum()))

        # num_bbox=feat_dict['num_bbox']
                
        #boxes spatial
        '''bbox=feat_dict['bbox']
        img_h=feat_dict['image_h']
        img_w=feat_dict['image_w']
        spatial_feat=get_spatial_feat(bbox,img_h,img_w) '''
            
        return img_feat


    def collate_fn(self, batches):
        img = None
        attribute = None
        bert_src = None

        src = [b['src'] for b in batches]
        oov_lists = [b['oov_list'] for b in batches]
        src_oov = [b['src_oov'] for b in batches]
        trg = [b['trg'] + [self.vocab('<eos>')] for b in batches]
        trg_class = [b['trg_class'] for b in batches]
        trg_oov = [b['trg_copy'] + [self.vocab('<eos>')] for b in batches]
        img_fns = [b['img'] for b in batches]

        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        original_indices = list(range(len(batches)))
        if self.use_type:
            types=[b['type'] for b in batches] 
        
            seq_pairs = sorted(
                zip(src, trg, trg_class, img_fns, types,trg_oov, src_oov, oov_lists, src_str, trg_str, original_indices),
                key=lambda p: len(p[0]), reverse=True)
            src, trg, trg_class, img_fns,types, trg_oov, src_oov, oov_lists, src_str, trg_str, original_indices = zip(*seq_pairs)
            src, src_lens, src_mask,type = self._pad(src,types)
        else:
               
            seq_pairs = sorted(
                zip(src, trg, trg_class, img_fns, trg_oov, src_oov, oov_lists, src_str, trg_str, original_indices),
                key=lambda p: len(p[0]), reverse=True)
            src, trg, trg_class, img_fns, trg_oov, src_oov, oov_lists, src_str, trg_str, original_indices = zip(*seq_pairs)
            src, src_lens, src_mask= self._pad(src)
            
        trg, trg_lens, trg_mask = self._pad(trg)
        trg_class = torch.LongTensor(trg_class)
        src_oov, _, _ = self._pad(src_oov)
        trg_oov, _, _ = self._pad(trg_oov)

        if self.use_img:
            if 0:
                imgs = []
                for img_fn in img_fns:
                    img_fn = img_fn.split('/')[-1].strip()
                    #img_line_id = self.url_map[img_fn]
                    img_feat=np.zeros((self.num_roi_boxes,self.img_feat_dim))
                    #print(self.imagefeat_dir,img_fn)
                    #print(img_fn)
                    a_img_feat=self.read_npz(self.imagefeat_dir, img_fn) 
                    #a_img_feat=self.read_npz('/home/dyfff/new/CMKP/data/CMKP_feature', img_fn) 
                    current_num=len(a_img_feat)
                    img_feat[:current_num,:]=a_img_feat
                    img_feat = torch.from_numpy(img_feat)
                    imgs.append(img_feat)
                    #img = torch.Tensor(self.img_feats[img_line_id])
                    #imgs.append(img)
                img = torch.stack(imgs, 0)
            else:
                imgs = []
                for img_fn in img_fns:
                    img_fn = img_fn.split('/')[-1].strip()
                    img_line_id = self.url_map[img_fn]
                    img = torch.Tensor(self.img_feats[img_line_id])
                    imgs.append(img)
                img = torch.stack(imgs, 0)

        if self.use_attribute:
            atts = []
            for img_fn in img_fns:
                print(img_fn)
                img_fn = img_fn.split('/')[-1].strip()
                img_line_id = self.url_map[img_fn]
                att = torch.Tensor(self.attribute_feats[img_line_id])
                atts.append(att)
            attribute = torch.stack(atts, 0)

        if self.use_bert_src:
            bert_srcs = []
            src_strs = [b['src_str'] for b in batches]
            for src_str in src_strs:
                src_str_id = self.src_str_map[src_str]
                bert_src = torch.Tensor(self.bert_feats[src_str_id])
                bert_srcs.append(bert_src)
            bert_src = torch.stack(bert_srcs, 0)

        if self.only_classifier:
            if self.use_type:
                return src, src_lens, src_mask, trg_class, img, attribute, bert_src,type
            return src, src_lens, src_mask, trg_class, img, attribute, bert_src

        # Yue: do not support bert feat for generator
        if self.is_test:
            if self.use_type:
                return src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, \
                   src_str, trg_str, original_indices, img, attribute,type
            return src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, \
                   src_str, trg_str, original_indices, img, attribute
        if self.use_type:
            return src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, img, attribute,type
        return src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, img, attribute

    def collate_itm(self, batches):
        img = None
        attribute = None
        bert_src = None

        src = [b['src'] for b in batches]
        oov_lists = [b['oov_list'] for b in batches]
        src_oov = [b['src_oov'] for b in batches]

        label=[b['label'] for b in batches]
        
        img_fns = [b['img'] for b in batches]

        src_str = [b['src_str'] for b in batches]

        original_indices = list(range(len(batches)))

        # sort all the sequences in the order of source lengths
        seq_pairs = sorted(
            zip(src, label, img_fns,  src_oov, oov_lists, src_str,  original_indices),
            key=lambda p: len(p[0]), reverse=True)
        src, label, img_fns, src_oov, oov_lists, src_str,  original_indices = zip(*seq_pairs)

        src, src_lens, src_mask = self._pad(src)
        
        src_oov, _, _ = self._pad(src_oov)
        

        if self.use_img:
            if 0:
                imgs = []
                for img_fn in img_fns:
                    img_fn = img_fn.split('/')[-1].strip()
                    #img_line_id = self.url_map[img_fn]
                    img_feat=np.zeros((self.num_roi_boxes,self.img_feat_dim))
                    #print(self.imagefeat_dir,img_fn)
                    #print(img_fn)
                    a_img_feat=self.read_npz(self.imagefeat_dir, img_fn) 
                    #a_img_feat=self.read_npz('/home/dyfff/new/CMKP/data/CMKP_feature', img_fn) 
                    current_num=len(a_img_feat)
                    img_feat[:current_num,:]=a_img_feat
                    img_feat = torch.from_numpy(img_feat)
                    imgs.append(img_feat)
                    #img = torch.Tensor(self.img_feats[img_line_id])
                    #imgs.append(img)
                img = torch.stack(imgs, 0)
            else:
                imgs = []
                for img_fn in img_fns:
                    try:
                        img_fn = img_fn.split('/')[-1].strip()
                        img_line_id = self.url_map[img_fn]
                        img = torch.Tensor(self.img_feats[img_line_id])
                        imgs.append(img)
                    except:
                        print("error",img_fn)
                img = torch.stack(imgs, 0)

        if self.use_attribute:
            atts = []
            for img_fn in img_fns:
                print(img_fn)
                img_fn = img_fn.split('/')[-1].strip()
                img_line_id = self.url_map[img_fn]
                att = torch.Tensor(self.attribute_feats[img_line_id])
                atts.append(att)
            attribute = torch.stack(atts, 0)

        if self.use_bert_src:
            bert_srcs = []
            src_strs = [b['src_str'] for b in batches]
            for src_str in src_strs:
                src_str_id = self.src_str_map[src_str]
                bert_src = torch.Tensor(self.bert_feats[src_str_id])
                bert_srcs.append(bert_src)
            bert_src = torch.stack(bert_srcs, 0)

        if self.only_classifier:
            return src, src_lens, src_mask, label, img, attribute, bert_src

       
        return src, src_lens, src_mask, src_oov, oov_lists, label, img, attribute
     


def get_tweet_img_loader(examples, vocab, trg_class_vocab,
                         use_text, use_img, use_attribute, use_type=None,use_bert_src=None,
                         img_feats_fn=None, attribute_feats_fn=None, url_map_fn=None,
                         bert_feats_fn=None, src_str_map_fn=None,
                         is_test=False, only_classifier=False, debug=False,
                         batch_size=16, shuffle=False, num_workers=4):
    multimodel_tweets = MultimodalTweetDataset(examples, vocab, trg_class_vocab,
                                               use_text, use_img, use_attribute,use_type, use_bert_src,
                                               img_feats_fn, attribute_feats_fn, url_map_fn,
                                               bert_feats_fn, src_str_map_fn,
                                               is_test, only_classifier, debug)

    data_loader = torch.utils.data.DataLoader(dataset=multimodel_tweets,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=multimodel_tweets.collate_fn)
    return data_loader

def get_itm_img_loader(examples, vocab, trg_class_vocab,
                         use_text, use_img, use_attribute, use_type=None,use_bert_src=None,
                         img_feats_fn=None, attribute_feats_fn=None, url_map_fn=None,
                         bert_feats_fn=None, src_str_map_fn=None,
                         is_test=False, only_classifier=False, debug=False,
                         batch_size=16, shuffle=False, num_workers=4):
    multimodel_tweets = MultimodalTweetDataset(examples, vocab, trg_class_vocab,
                                               use_text, use_img, use_attribute,use_type,use_bert_src,
                                               img_feats_fn, attribute_feats_fn, url_map_fn,
                                               bert_feats_fn, src_str_map_fn,
                                               is_test, only_classifier, debug)

    data_loader = torch.utils.data.DataLoader(dataset=multimodel_tweets,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=multimodel_tweets.collate_itm)
    return data_loader