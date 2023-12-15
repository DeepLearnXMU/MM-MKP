import re
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import time


def get_text_processor():
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "elongated", "repeated", 'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    return text_processor


def write_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def tw_tokenize(src_fn, trg_fn, wiki_fn,ocr_dict, valid_token_set):
    text_processor = get_text_processor()
    t0 = time.time()
    def process_sentence(text, text_processor=text_processor):
        # return a list of tokens
        tokens = text_processor.pre_process_doc(text)
        filter_set = {'<repeated>', '<elongated>'}
        return [t for t in tokens if t not in filter_set]

    valid_idx = 0
    fw = open(trg_fn, 'w', encoding='utf-8')
    with open(src_fn, 'r', encoding='utf-8') as fr,open(wiki_fn, 'r', encoding='utf-8') as fr2:
        fr_list=fr.readlines()
        fr2_list=fr2.readlines()
    for idx, (line,line2) in enumerate(zip(fr_list,fr2_list)):
        if 1:
            #print(line)
            #print(line2)
            img_fn = line.split('/')[-1].strip()
            
            text=line.split('<sep>')[0].strip()
            kw=line.split('<sep>')[1].strip()
            jpg_url=line.split('<sep>')[-1].strip()
            img_fn2=jpg_url.split('/')[-1].strip()
            assert img_fn==img_fn2
            itm_pro=line2.split('<sep>')[3].strip()
            itm=line2.split('<sep>')[4].strip()
            wiki=line2.split('<sep>')[-1].strip()

            
            
            wiki_list=wiki.split(';')
            wiki=' ; '.join(wiki_list)
            #print(wiki)
            
            
            if img_fn not in ocr_dict:
                ocr=' '
                res=text+' <seg> '+ocr+' <seg> '+wiki+'<sep>'+kw+'<sep>'+jpg_url+'<sep>'+itm_pro+'<sep>'+itm+'<sep>'+wiki
                #print("res",res)
                #assert 1==0
                fw.write(res+'\n')
                continue
            ocr_text = ocr_dict[img_fn]
            tokens = process_sentence(ocr_text)
            tokens = list(filter(lambda x: re.match('^[a-zA-Z]+$', x), tokens))
            tokens = list(filter(lambda x: x in valid_token_set and len(x) > 1, tokens))

            long_num = sum([len(t) > 3 for t in tokens])
            short_num = sum([len(t) < 3 for t in tokens])
            if len(tokens) == 0 or short_num / len(tokens) >= 0.75 or long_num < 3:
                ocr=' '
                res=text+' <seg> '+ocr+' <seg> '+wiki+' <sep>'+kw+' <sep>'+jpg_url+'<sep>'+itm_pro+'<sep>'+itm+'<sep>'+wiki
                #print("res",res)
                #assert 1==0
                fw.write(res+'\n')
            else:
                ocr=' '.join(tokens)
                res=text+' <seg> '+ocr+' <seg> '+wiki+' <sep>'+kw+'<sep>'+jpg_url+'<sep>'+itm_pro+'<sep>'+itm+'<sep>'+wiki
                #print("res",res)
                #assert 1==0
                fw.write(res+'\n')
                valid_idx += 1
        idx += 1
    fw.close()
    print('Writing %d tweets into %s, takes %.2f seconds' % (idx, trg_fn, time.time() - t0))
    print('Valid OCR rate is %d/%d = %.2f' % (valid_idx, idx, valid_idx / idx))


if __name__ == '__main__':
    t0 = time.time()
    ocr_fn = '/home/dyfff/log/ocr_last2.txt'
    src_fn = '/home/dyfff/new/CMKP/data/tw_mm_s1/{}_src.txt'
    trg_fn = '/home/dyfff/new/CMKP/data/tw_mm_s1_ocr/{}_ocr_kw.txt'
    wiki_fn='/home/dyfff/CMKP/data/tw_mm_s1_ocr/{}_src_kw.txt'

    all_tokens = []
    for tag in ['train', 'valid', 'test']:
        src_lines = open(src_fn.format(tag), 'r').readlines()
        for line in src_lines:

            #rip pop . such a shame this team is the way they are .<sep>['steelers']<sep>/home/dyfff/CMKP/data/CMKP_images/Dv7wlFiUUAANZP1.jpg<sep>tensor([[9.9992e-01, 7.6948e-05]])<sep>0<sep>0.676886##Arsene Wenger##Arsène Charles Ernest Wenger OBE (French pronunciation: ​[aʁsɛn vɛŋɡɛʁ]; born 22 October 1949) is a French former football manager and player who is currently&&0.521176##Cartoon cartoon characters##Cartoon Network (often abbreviated as CN) is an American cable television channel owned by Warner Bros. Discovery. It is a part of The Cartoon Network

            tags = line.strip().split('<sep>')[1].strip().split(';')
            tag_tokens = []
            for tag in tags:
                tag_tokens.extend(tag.split())
            post_tokens = line.strip().split('<sep>')[0].split()
            all_tokens += tag_tokens + post_tokens

    valid_token_set = set(all_tokens)
    print('The size of valid token set: %d, takes %.2f seconds' % (len(valid_token_set), time.time() - t0))

    ocr_dict = dict()
    with open(ocr_fn, 'r') as f:
        for idx, line in enumerate(f):
            sep_id = line.index(':')
            k = line[:sep_id].strip()
            v = line[sep_id + 1:].strip().replace('\n', '').replace('\r', '')
            if len(v) != 0:
                ocr_dict[k] = v
    print('There are %d/%d in %s has OCR, takes %.2f seconds' % (len(ocr_dict), idx + 1, ocr_fn, time.time() - t0))
    # There are 49621/146656 in CMKP_ocr.txt has OCR, takes 3.27 seconds

    for tag in ['train', 'valid', 'test']:
        cur_src_fn = src_fn.format(tag)
        cur_trg_fn = trg_fn.format(tag)
        cur_wiki_fn=wiki_fn.format(tag)
        tw_tokenize(cur_src_fn, cur_trg_fn,cur_wiki_fn, ocr_dict, valid_token_set)
