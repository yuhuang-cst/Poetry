# encoding: UTF-8

"""
@author: hy
"""

from Config import *
from Common import *
import numpy as np
import random

class NMTReader(object):
	def __init__(self, config, bcInitializer):
		self.config = config
		self.vocabList = self.createOrReadVocab(VOCAB_PATH, [TRAIN_DATA_PATH, VALID_DATA_PATH], self.config.vocabSize)  # [word1, word2, ...]
		self.wordToId = {self.vocabList[id]: id for id in range(len(self.vocabList))}
		self.train = bcInitializer(TRAIN_DATA_PATH, self)
		self.valid = bcInitializer(VALID_DATA_PATH, self)
		self.test = bcInitializer(TEST_DATA_PATH, self)


	def idsToSen(self, idList):
		"""
		Args:
			idList (list): [id1, id2, ...], type(id)=int
		Returns:
			unicode: sentence
		"""
		return ''.join([self.vocabList[id] for id in idList])


	def senToIds(self, sen):
		"""
		Args:
			sen (unicode): sentence
		Returns:
			list: [id1, id2, ...], type(id)=int
		"""
		return [self.wordToId[w] if w in self.wordToId else self.wordToId[UNK] for w in sen]


	def createOrReadVocab(self, vocabPath, srcPathList, vocabSize=None):
		"""
		Returns:
			list: [word1, word2, ...]
		"""
		if os.path.exists(vocabPath):
			with open(vocabPath) as f:
				charList = f.read().split('\n')
				if vocabSize == None or len(charList) == vocabSize:
					return charList
				else:
					print('size of vocab in file is not equal to the given vocab')
		print('creating vocab file...')
		counter = Counter()
		for filepath in srcPathList:
			with open(filepath) as f:
				data = re.sub('\s', '', f.read())
				counter.update(data)
		if vocabSize == None:
			vocabSize = len(counter)
		assert vocabSize <= len(counter)
		wordFreqs = sorted(counter.items(), key=lambda item: item[1], reverse=True)[:vocabSize-3]
		vocab = [EOS, SOS, UNK] + [word for word, freq in wordFreqs]
		with open(vocabPath, 'w') as f:
			print('total vocab number=%d, keep number=%d' % (len(counter), len(vocab)))
			f.writelines('\n'.join(vocab))
		return vocab


class BatchController(object):
	def __init__(self, filepath, reader):
		self.contextReader = reader
		self.wordToId = reader.wordToId
		self.readData(filepath)
		self.currentRank = 0
		self.dataPadding = None


	def readData(self, filepath):
		self.dataRaw = readPair(filepath)  # [(FS, SS), ...], type(FS)=type(SS)=unicode
		self.filterDataRaw()
		self.dataSize = len(self.dataRaw)
		self.data = [(self.senToIds(FS), self.senToIds(SS)) for FS, SS in self.dataRaw]  # [([id1, id2, ...], [id3, id4, ...]), ...]
		self.posList = list(range(len(self.data)))


	def filterDataRaw(self):
		pass


	def senToIds(self, sen):
		return self.contextReader.senToIds(sen)


	def padding(self, seqIds, paddingId, maxTime=None):
		"""
		Args:
			seqIds (list of list): [[id1, id2, ...], ...], shape=(batchSize, None)
			paddingId (int): the id using for padding
		Returns:
			np.ndarray: padding seqIds, shape=(batchSize, maxTime)
			np.ndarray: seqLength, shape=(batchSize)
			int: maxTime
		"""
		seqLength = np.array([len(idList) for idList in seqIds], dtype=np.int32)
		if maxTime == None:
			maxTime = max(seqLength)
		newSeqIds = np.array([idList+[paddingId]*(maxTime-len(idList)) for idList in seqIds], dtype=np.int32)
		return newSeqIds, seqLength, maxTime


	def paddingBatch(self, batch, paddingId=0, maxTime=None):
		"""
		Args:
			batch (list): [(FS, SS), ...], FS=[id1, id2, ...], SS=[id3, id4, ...]
		Returns:
			numpy.array: srcSeqIds, shape=(batchSize, srcMaxTime)
			numpy.array: srcSeqLength, shape=(batchSize), 有效长度
			numpy.array: tgtSeqIds, shape=(batchSize, tgtMaxTime)
			numpy.array: tgtSeqLength, shape=(batchSize), 有效长度
		"""
		srcSeqIds, tgtSeqIds = zip(*batch)  # src=[[id1, id2, ...], ...]; tgt=[[id3, id4, ...], ...]
		srcSeqIds, srcSeqLength, _ = self.padding(srcSeqIds, paddingId, maxTime)
		tgtSeqIds, tgtSeqLength, _ = self.padding(tgtSeqIds, paddingId, maxTime)
		return srcSeqIds, srcSeqLength, tgtSeqIds, tgtSeqLength


	def nextBatch(self, batchSize):
		raise NotImplementedError


	def all(self):
		if self.dataPadding == None:
			self.dataPadding = self.paddingBatch(self.data)
		return self.dataPadding


	def allDataRaw(self):
		return [self.dataRaw[pos] for pos in self.posList]



class Seq2seqBatchController(BatchController):
	def __init__(self, filepath, reader):
		super(Seq2seqBatchController, self).__init__(filepath, reader)


	def paddingBatch(self, batch, paddingId=0, maxTime=None):
		"""
		Args:
			batch (list): [(FS, SS), ...], FS=[id1, id2, ...], SS=[id3, id4, ...]
		Returns:
			numpy.array: srcSeqIds, shape=(batchSize, srcMaxTime)
			numpy.array: srcSeqLength, shape=(batchSize), 有效长度
			numpy.array: tgtInputIds, shape=(batchSize, tgtMaxTime)
			numpy.array: tgtOutputIds, shape=(batchSize, tgtMaxTime)
			numpy.array: tgtSeqLength, shape=(batchSize), 有效长度
		"""
		srcSeqIds, tgtSeqIds = zip(*batch)  # src=[[id1, id2, ...], ...]; tgt=[[id3, id4, ...], ...]
		tgtInputIds = [[self.wordToId[SOS]]+idList for idList in tgtSeqIds]
		tgtOutputIds = [idList+[self.wordToId[EOS]] for idList in tgtSeqIds]
		srcSeqIds, srcSeqLength, _ = self.padding(srcSeqIds, paddingId)
		tgtInputIds, tgtSeqLength, _ = self.padding(tgtInputIds, paddingId)
		tgtOutputIds, _, _ = self.padding(tgtOutputIds, paddingId)
		return srcSeqIds, srcSeqLength, tgtInputIds, tgtOutputIds, tgtSeqLength


	def nextBatch(self, batchSize, srcReverse=False):
		"""
		Args:
			batchSize (int): batchSize
		Returns:
			numpy.array: srcSeqIds, shape=(batchSize, srcMaxTime)
			numpy.array: srcSeqLength, shape=(batchSize), 有效长度
			numpy.array: tgtInputIds, shape=(batchSize, tgtMaxTime)
			numpy.array: tgtOutputIds, shape=(batchSize, tgtMaxTime)
			numpy.array: tgtSeqLength, shape=(batchSize), 有效长度
			int: number of element not fetched yet
		"""
		if self.dataSize - self.currentRank < batchSize:
			random.shuffle(self.posList)   # 洗牌
			self.currentRank = 0
		batch = [self.data[pos] for pos in self.posList[self.currentRank: self.currentRank+batchSize]]  # [([id1, id2, ...], [id3, id3, ...]), ...]
		srcSeqIds, srcSeqLength, tgtInputIds, tgtOutputIds, tgtSeqLength = self.paddingBatch(batch)
		if srcReverse:
			srcSeqIds = srcSeqIds[:, ::-1]
		self.currentRank += batchSize
		return srcSeqIds, srcSeqLength, tgtInputIds, tgtOutputIds, tgtSeqLength, self.dataSize-self.currentRank #self.paddingBatch(batch)


class NaiveBatchController(BatchController):
	def __init__(self, filepath, reader):
		super(NaiveBatchController, self).__init__(filepath, reader)


	def nextBatch(self, batchSize):
		"""
		Args:
			batchSize (int): batchSize
		Returns:
			numpy.array: srcSeqIds, shape=(batchSize, srcMaxTime)
			numpy.array: tgtSeqIds, shape=(batchSize, tgtMaxTime)
			numpy.array: seqLength, shape=(batchSize), 有效长度
			int: number of element not fetched yet
		"""
		if self.dataSize - self.currentRank < batchSize:
			random.shuffle(self.posList)   # 洗牌
			self.currentRank = 0
		batch = [self.data[pos] for pos in self.posList[self.currentRank: self.currentRank+batchSize]]  # [([id1, id2, ...], [id3, id3, ...]), ...]
		srcSeqIds, srcSeqLength, tgtSeqIds, tgtSeqLength = self.paddingBatch(batch)
		self.currentRank += batchSize
		return srcSeqIds, tgtSeqIds, srcSeqLength, self.dataSize-self.currentRank #self.paddingBatch(batch)


if __name__ == '__main__':
	nmtReader = NMTReader(NMTSeq2seqConfig(), Seq2seqBatchController)
	for i in range(10):
		srcSeqIds, srcSeqLength, tgtInputIds, tgtOutputIds, tgtSeqLength, _ = nmtReader.train.nextBatch(20, True)
		print('---------------------------\n')
		print('srcSeqIds', type(srcSeqIds), srcSeqIds.shape, srcSeqIds[0])
		print('srcSeqLength', type(srcSeqLength), srcSeqLength.shape, srcSeqLength[0])
		print('tgtInputIds', type(tgtInputIds), tgtInputIds.shape, tgtInputIds[0])
		print('tgtOutputIds', type(tgtOutputIds), tgtOutputIds.shape, tgtOutputIds[0])
		print('tgtSeqLength', type(tgtSeqLength), tgtSeqLength.shape, tgtSeqLength[0])

	# nmtReader = NMTReader(NMTSeq2seqConfig(), NaiveBatchController)
	# for i in range(10):
	# 	srcSeqIds, tgtSeqIds, seqLength, _ = nmtReader.train.nextBatch(20)
	# 	print('---------------------------\n')
	# 	print('srcSeqIds', type(srcSeqIds), srcSeqIds.shape, srcSeqIds[0])
	# 	print('seqLength', type(seqLength), seqLength.shape, seqLength[0])
	# 	print('tgtSeqIds', type(tgtSeqIds), tgtSeqIds.shape, tgtSeqIds[0])




