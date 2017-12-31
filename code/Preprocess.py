# encoding: UTF-8

"""
@author: hy
"""

from Config import *
import re
from sklearn.model_selection import train_test_split
from Common import *


class Preprocessor(object):
	def __init__(self):
		self.emp = re.compile('\s')
		pass

	def washEmpty(self, s):
		return self.emp.sub('', s)


	def isCNS(self, c):
		return '\u4e00' <= c <= '\u9fff'


	def isCNSSen(self, s):
		for c in s:
			if not self.isCNS(c):
				return False
		return True


	def readData(self, filepath):
		raise NotImplementedError


class QTSPreprocessor(Preprocessor):
	def __init__(self):
		super(QTSPreprocessor, self).__init__()
		self.poemMa = re.compile('^.*?〖(.*?)〗.*?作者:(.*?)\n(.*?)$', re.S)
		self.poemSub = re.compile('\(.*?\)|（.*?）')


	def handleSingle(self, poem):
		"""
		Args:
			poem (unicode): raw poem
		Returns:
			unicode: poem
			unicode: title
			unicode: author
		"""
		m = self.poemMa.match(poem)
		if not m:
			return None, None, None
		title, author, poem = m.group(1), m.group(2), m.group(3)
		poem = self.poemSub.sub('', poem)
		return self.washEmpty(poem), self.washEmpty(title), self.washEmpty(author)


	def readData(self, filepath):
		"""读取诗歌, 去重
		Args:
			filepath (str): file path
		Returns:
			list: [poem1, poem2, ...], type(poem)==unicode
		"""
		data = open(filepath).read()    #unicode
		rawList = re.split('<http.*?>', data)
		poemSet = set()
		for rawPoem in rawList:
			poem, _, _ = self.handleSingle(rawPoem)
			if poem != None:
				poemSet.add(poem)
		return list(poemSet)


	def getPair(self, poemList):
		pairList = []  # [(FS, SS)], FS=First Sentence, SS=Second Sentence
		otherSen = [] # [sen1, sen2]
		for poem in poemList:
			senList = poem.split('。')
			for sen in senList:
				pair = sen.split('，')
				if len(pair) == 2 and len(pair[0]) == len(pair[1]):
					pairList.append(tuple(pair))
				else:
					for s in pair:
						if len(s) != 0 and self.isCNSSen(s):
							otherSen.append(s)
		return pairList, otherSen


	def savePair(self, pairList, filepath):
		with open(filepath, 'w') as f:
			for i in range(len(pairList)-1):
				FS, SS = pairList[i]
				f.write('%s %s\n' % (FS, SS))
			FS, SS = pairList[-1]
			f.write('%s %s' % (FS, SS))


	def saveSen(self, senList, filepath):
		with open(filepath, 'w') as f:
			for i in range(len(senList) - 1):
				f.write((senList[i] + '\n'))
			f.write(senList[-1])


def printStatistic(senList):
	V, N, _ = countCharacter(senList)
	return 'character number(V, 去重)=%d, character number(N, 未去重)=%d' % (V, N)


if __name__ == '__main__':
	proprocessor = QTSPreprocessor()

	poemList = proprocessor.readData(DATA_PATH + '/raw/全唐诗.txt')
	print('poem number=%d,' % (len(poemList)), printStatistic([re.sub('。|，', '', poem) for poem in poemList]))

	pairList, otherSen = proprocessor.getPair(poemList) #lmData: language model Data
	print('all pair number=%d,' % (len(pairList)), printStatistic([FS+SS for FS, SS in pairList]))

	train, test = train_test_split(pairList, test_size=TEST_PROPORTION)
	train, validate = train_test_split(train, test_size=VALIDATE_PROPORTION)
	print('train pair number=%d,' % (len(train)), printStatistic([FS+SS for FS, SS in train]))
	print('validation pair number=%d,' % (len(validate)), printStatistic([FS+SS for FS, SS in validate]))
	print('test pair number=%d,' % (len(test)), printStatistic([FS+SS for FS, SS in test]))

	lmSenData = otherSen + [sen for pair in train for sen in pair]
	print('lm sentence number=%d,' % (len(lmSenData)), printStatistic(lmSenData))

	proprocessor.savePair(train, TRAIN_DATA_PATH)
	proprocessor.savePair(validate, VALID_DATA_PATH)
	proprocessor.savePair(test, TEST_DATA_PATH)
	proprocessor.saveSen(lmSenData, LM_DATA_PATH)












