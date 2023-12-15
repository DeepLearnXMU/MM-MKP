import os
import pickle
import csv
if __name__ == '__main__':
    data_tag = 'tw_mm_s1_ocr'  # 'tw_mm_s1' || 'tw_mm_imagenet_s2' || 'tw_mm_daily_s2'
    data_dir = '../data/{}'.format(data_tag)
    split=3576
    img_fn = '/home/dyfff/new/CMKP/data/textimage-data.csv'
    with open(img_fn,'r',  encoding='utf-8') as csv_file:
   
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
        imgs = []
        
        for row in csv_reader:
            img=f"T{row['tweet_id']}.jpg"
            imgs.append(img)
    train_imgs=imgs[:split]
    test_imgs=imgs[split:]


    for data_tag in ['train', 'test']:
        print('\nComputing url map for %s' % data_tag)
        
        trg_fn = os.path.join(data_dir, '{}_itm_url_map.pt'.format(data_tag))
        url_map = {}
        if data_tag=='train':
            data_imgs=train_imgs
        elif data_tag=='test':
            data_imgs=test_imgs
        print(len(data_imgs))
        #with open(src_fn, 'r', encoding='utf-8') as fr:
        
        for idx, img_fn in enumerate(data_imgs):
            
            if img_fn not in url_map.keys():
                url_map[img_fn] = idx
            else:
                print('Error, there are duplicate img filenames: %s' % img_fn)
        
        with open(trg_fn, 'wb') as fw:
            pickle.dump(url_map, fw)
        print('Dump %d items of a dict into %s' % (len(url_map), trg_fn))
