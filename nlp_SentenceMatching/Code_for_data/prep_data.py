# coding:utf-8

import numpy as np
import csv

import collections


class PrepData():
    def __init__(   self,
                    wordsNum=80000,
                    dataDir=str('data/train_dir/'),
                    saveFileName=str('saveData.tsv'),
                    saveDictPath=str('word_dict.txt')
                ):
        # self.data_path = dataPath
        # self.save_path = savePath
        self._idx2vocab = ['PAD', 'UNK']
        self.PAD = 0
        self.UNK = 1
        self._vocab2idx = {}

        self.DataDir = dataDir
        self.saveFileName = saveFileName
        self.saveDictPath = saveDictPath

        self.num_category = 2
        self.wordsNum = wordsNum

    def dataPreparation(self, dataPath, low_frequency=10 ):
        savePath = self.saveFileName
        saveDictPath = self.saveDictPath
        self.readCsvData( dataPath, savePath )
        self.saveWordDict( savePath, saveDictPath)
        # get word_dict fiter low frequency word, and set the self._vocab2idx  to encoder seq_str to label_list;
        word_dict = self.__getWrodDict( saveDictPath )
        self.filterWordDict( word_dict, lowFreq=low_frequency )

    def getBatchData(self, batch_size=128):
        # 只利用 yield 对大文件，进行切片读取。逐行读取，便于对 每行 进行预处理（包括分词，去停用词，各行 分列 处理等 ）。
        savePath = self.saveFileName
        def _normalize_length(_data, max_length):
            return _data + [ self.PAD ] * ( max_length - len(_data)) 
        seq1_data, seq1_lengths, seq2_data, seq2_lengths, labels = [], [], [], [], []
        with open( savePath, 'r' ) as csvfile:
            csvReader = csv.DictReader(csvfile, delimiter='\t')
            index = 0
            for row in csvReader:
                index +=1
                # self.__encoder( seq_str ) # transfer the seq_str to label_list( like a Interger list )
                q1_seq, q2_seq, label = self.__encoder(row['q1']), self.__encoder(row['q2']), row['label']
                seq1_data.append( q1_seq )
                seq1_lengths.append( len(q1_seq) )
                seq2_data.append( q2_seq )
                seq2_lengths.append( len(q2_seq) )
                labels.append( label )
                if index % batch_size ==0:
                    seq1_max_length = max(seq1_lengths)
                    seq2_max_length = max(seq2_lengths)
                    seq1_data = list(map( lambda item: _normalize_length(item, seq1_max_length), seq1_data ) ) 
                    seq2_data = list(map( lambda item: _normalize_length(item, seq2_max_length), seq2_data ) )
                    seq1_data = [ list(map(int, item_list)) for item_list in seq1_data ]
                    seq2_data = [ list(map(int, item_list)) for item_list in seq2_data ]
                    batch_data_dict = {     'sentence1_inputs': np.asarray(seq1_data, dtype=np.int32),
                                            'sentence1_lengths': np.asarray(seq1_lengths, dtype=np.int32),
                                            'sentence2_inputs': np.asarray(seq2_data, dtype=np.int32),
                                            'sentence2_lengths': np.asarray(seq2_lengths, dtype=np.int32),
                                            'labels': np.asarray(labels, dtype=np.int32)
                                      }
                    yield batch_data_dict
                    seq1_data, seq1_lengths, seq2_data, seq2_lengths, labels = [], [], [], [], []
            if len( labels ) != 0:
                seq1_max_length = max(seq1_lengths)
                seq2_max_length = max(seq2_lengths)
                seq1_data = map( lambda item: _normalize_length(item, seq1_max_length), seq1_data )
                seq2_data = map( lambda item: _normalize_length(item, seq2_max_length), seq2_data )
                seq1_data = [ list(map(int, item_list)) for item_list in seq1_data ]
                seq2_data = [ list(map(int, item_list)) for item_list in seq2_data ]
                batch_data_dict = {     'sentence1_inputs': np.asarray(seq1_data, dtype=np.int32),
                                        'sentence1_lengths': np.asarray(seq1_lengths, dtype=np.int32),
                                        'sentence2_inputs': np.asarray(seq2_data, dtype=np.int32),
                                        'sentence2_lengths': np.asarray(seq2_lengths, dtype=np.int32),
                                        'labels': np.asarray(labels, dtype=np.int32)
                                    }
                yield batch_data_dict
        # end


    def saveWordDict(self, dataPath, dictPath):
        assert dataPath, "dataPath is None"
        assert dictPath, "dictPath is None"
        self.dictPath= dictPath
        word_dict = collections.defaultdict(int)
        with open( dataPath, 'r' ) as csvfile:
            csvReader = csv.DictReader(csvfile, delimiter='\t')
            sample_num = 0
            for row in csvReader:
                sample_num +=1
                q1_list = row['q1'].strip().split()
                q2_list = row['q2'].strip().split()
                for word in q1_list + q2_list:
                    word_dict[word] +=1
        with open( dictPath, 'w' ) as dictFile:
            for key,val in word_dict.items():
                dictFile.write( str(key) +" "+str(val) +"\n" )
        self.sample_num = sample_num  #  data数据 样本大小；
        print( "save wordDict done... Path:{}; WordDict size:{}".format( dictPath, len(word_dict) ) )
    
    # 
    def filterWordDict(self, word_dict, lowFreq=10):
        new_wordDict = {  }
        for word,count in word_dict.items():
            if count <= lowFreq:
                continue
            new_wordDict[word] = count
        new_wordDict_list = sorted(new_wordDict.items(), key=lambda x: (x[1], x[0]), reverse=True)
        self._idx2vocab.extend( [word for word,count in new_wordDict_list] )  # 用语料库的词  扩展 idx2vocab 表。
        
        if len(self._idx2vocab) >= self.wordsNum:
            self._idx2vocab = self._idx2vocab[: self.wordsNum]
        else:
            self.wordsNum = len( self._idx2vocab )

        self._vocab2idx = {v:i for i,v in enumerate(self._idx2vocab)}  # 由 word 得到 idx
        new_dictSize = len( self._vocab2idx )
        print( "new wordDict size:{}; words num:{}".format( new_dictSize, self.wordsNum ) )
        return new_wordDict, new_dictSize
    
    def __encoder(self, seq_str):  # 将 词序列， 转换成 idx 序列（ list 形式）；
        words = seq_str.split()
        return [self._vocab2idx[word] if word in self._vocab2idx else self.UNK for word in words]

    def __getWrodDict(self, dictPath):
        word_dict = {  }
        with open(self.dictPath, 'r') as dictFile:
            for line in dictFile.readlines():
                line_list = line.rstrip().split()
                # assert len(line_list)!=2
                if len(line_list) !=2 :
                    print( line_list )
                    continue
                word = line_list[0]
                count = int( line_list[1] )
                word_dict[word] = count
        return word_dict

    
    # 逐行读取数据，并做初步的筛选 特殊字符；   可另加其他操作，如：分词，去停用词，词性替换，同义词替换，等等；
    def readCsvData( self, dataPath, savePath ):
        assert dataPath, "dataPath is None~"
        assert savePath, "savePath is None"
        self.dataPath = dataPath
        self.savePath = savePath
        with open( savePath, 'w' ) as writerFile:
            fieldnames = [ 'q1', 'q2', 'label' ]
            writer = csv.DictWriter( writerFile, delimiter='\t', fieldnames=fieldnames )
            writer.writeheader()
            label_dict = collections.defaultdict(int)
            with open( dataPath, 'r' ) as csvfile:
                csvReader = csv.DictReader(csvfile, delimiter='\t')
                for row in csvReader:
                    q1 = self.__handler( row['question1']  )
                    q2 = self.__handler( row['question2'] )
                    label = row['is_duplicate']
                    label_dict[label] +=1
                    writer.writerow({'q1': q1, 'q2': q2, 'label': label })
        
        self.num_category = len( label_dict )
        print( "read and write done...  num_category size:{}".format( self.num_category ) )


    def __handler(self, ques_str):
        ques_str = str(ques_str)
        stop_char_list = [ ',', '.', '?', '"', '/', "'", '(', ')' ]
        # stop_word = {  }
        for char in stop_char_list:
            ques_str = ques_str.replace( char, '' )
        return ques_str






