# coding:utf-8

from itertools import islice

# 参考  https://stackoverflow.com/questions/39549426/read-multiple-lines-from-a-file-batch-by-batch

class DataProp():
    def __init__(self, data_path, batch_size=128):
        self.file_path = data_path
        self.batch_size = batch_size

    # 利用 islice 对文件切片 分割； 代码简洁； 也有一个问题，就是 返回的迭代器，每行为一个元素，无法对每行中的 各列 进行操作。 较为繁琐。
    def getBatchData(self):
        with open( self.file_path, 'r' ) as data_file:
            self.batch_num = 0
            for n_lines in iter(lambda: tuple( islice(data_file, self.batch_size) ), ()):
                self.batch_num +=1
                yield n_lines

    # 只利用 yield 对大文件，进行切片读取。逐行读取，便于对 每行 进行预处理（包括分词，去停用词，各行 分列 处理等 ）。
    def getChunkData(self):
        with open( self.file_path, 'r' ) as data_file:
            chunk_list = []
            index = 0
            for line in data_file:
                index +=1
                chunk_list.append( line )   # line 可以是处理后的结果，line.rstrip().split('\t)  分列处理后的 各列等。 利用 np.asarray() 转换为 DP能利用的形式。
                if index%self.batch_size ==0:
                    yield chunk_list
                    chunk_list = []
            if len(chunk_list) !=0:  #  最后一个 chunk sample 个数不够 batch_size，且不为 0 时。
                yield chunk_list



if __name__=='__main__':
    file_path = str('quora_duplicate_questions.tsv')
    batch_size = 128
    DataHelper = DataProp( file_path, batch_size=batch_size )
    for data in DataHelper.getChunkData():
        print( len(data) )
    
