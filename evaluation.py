
import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
from PIL import Image

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start,step,TP,P,T,input_type,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                if num_cls == 81:
                    predict = predict - 91
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((num_cls,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
                # 先判断

            if ('COCO' in str(gt_folder) or 'coco' in str(gt_folder)):
                gt_file = os.path.join(gt_folder,'%s.png'%name[15:27])
            else:
                gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    # loglist_ = []
    for i in range(num_cls):
        loglist[categories[i]] = round(IoU[i] * 100, 1)
        # loglist_.append(round(IoU[i] * 100, 3))
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = round(miou * 100, 2)
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = round(fp * 100, 2)
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = round(fn * 100, 2)
    if printlog:
        for i in range(num_cls):
            # print(print('%11s:%7.3f%%'%(categories[i],IoU[i]*100)))
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writedict(file, dictionary, curve=True):
    if curve:
        s = ''
        for key in dictionary.keys():
            # sub = '%s:%s  '%(key, dictionary[key])
            sub = '%s,'%(dictionary[key])
            s += sub
        s += '\n'
        file.write(s)
    else:
        str_format = "{:<15s}\t{:<15.2f}"
        idx = 0
        # breakpoint()
        # with open(filepath, 'a') as f:
        for k in dictionary.keys():
            if not isinstance (dictionary[k], list):
                print(str_format.format(k, dictionary[k]))
                file.write('class {:2d} {:12} IU {:.3f}'.format(idx, k, dictionary[k]) + '\n')
                idx += 1
            else:
                s=''
                for item in dictionary[k]:
                    s += '%s,'%(item)
                s += '\n'
                file.write(s)
    # file.write('mIoU = {:.3f}'.format(mIoU) + '\n')
    # print(f'mIoU={mIoU:.3f}')

def writelog(filepath, metric, comment, curve=True):
    print(filepath)
    filepath = filepath
    if os.path.exists(os.path.dirname(filepath)):
        pass
    else:
        os.makedirs(os.path.dirname(filepath))
    
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric, curve)
    logfile.write('=====================================\n')
    logfile.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt',type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=0.37, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--start', default=32, type=int)
    parser.add_argument('--end', default=42, type=int)
    args = parser.parse_args()
    
    # breakpoint()
    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    
    # 只评价 single_class 的图片
    sing_id_txt = open("voc12/train_aug_id.txt")
    single_idx = []
    all_lines = sing_id_txt.readlines()
    for line in all_lines:
        single_idx.append(line.strip())
    sing_id_txt.close()
    
    # 取二者的交集
    # breakpoint()
    name_list = list(set(name_list) & set(single_idx))
    print('Is evaluating {} images.'.format(len(name_list)))
    # breakpoint()
    args.curve = False
    if 'coco2014' in str(args.list):
        name_list = name_list[:5000]   # 如果是coco2014只评测前5k张图片
    if not args.curve:
        for t in [0.38]:
            args.t = t
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True)
            writelog(args.logfile, loglist, args.comment, curve=False)
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        for i in range(args.start, args.end):
            # print(i, args.start, args.end)
            t = i/100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, t)
            l.append(loglist['mIoU'])
            print('%d/50 background score: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            # else:
            #     break
        print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment, curve=True)

