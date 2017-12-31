# encoding: UTF-8

"""
@author: hy
"""

from collections import Counter
import re
import os


def readLine(filepath):
	"""
	Args:
		filepath (str): path
	Returns:
		list: [sen1, sen2, ...], type(sen)=unicode
	"""
	with open(filepath) as f:
		return f.read().split('\n')


def readPair(filepath):
	"""
	Args:
		filepath (str)
	Returns:
		list: [[FS, SS], ...]
	"""
	pairList = readLine(filepath)
	for i in range(len(pairList)):
		pairList[i] = pairList[i].split(' ')
	return pairList


def saveResult(pairList, TSSList, filepath):
	"""
	Args:
		pairList (list): [(FS, SS), ...], type(FS) = type(FS) = unicode
		TSSList (list): [TSS, ...], type(TSS) = unicode, TSS=Target Second Sentence
		filepath (str): file path
	"""
	with open(filepath, 'w') as f:
		for i in range(len(pairList)):
			line = '%s %s |Ref:%s\n' % (pairList[i][0], TSSList[i], pairList[i][1])
			f.write(line)


def savePair(pairList, filepath):
	with open(filepath, 'w') as f:
		for i in range(len(pairList)):
			line = '%s %s\n' % (pairList[i][0], pairList[i][1])
			f.write(line)


def readData(trainPath, validationPath, testPath, lmPath):
	print('Reading data...')
	data = {}
	data['train'] = readPair(trainPath)
	data['validation'] = readPair(validationPath)
	data['test'] = readPair(testPath)
	data['lm'] = readLine(lmPath)
	return data


def countCharacter(senList):
	"""
	Args:
		senList (list): [unicode]
	Returns:
		int, int, Counter: V(去重), N(未去重), Counter dict
	"""
	charCounter = Counter()
	for s in senList:
		charCounter.update(s)
	return len(charCounter), sum(charCounter.values()), charCounter


def keyWrap(k, d, subKey):
	return k if k in d else subKey


def idsToSen(idList, vocabList):
	"""
	Args:
		idList (list): [id1, id2, ...], type(id)=int
	Returns:
		unicode: sentence
	"""
	return ''.join([vocabList[id] for id in idList])


def senToIds(sen, wordToId):
	"""
	Args:
		sen (unicode): sentence
	Returns:
		list: [id1, id2, ...], type(id)=int
	"""
	return [wordToId[w] if w in wordToId else wordToId[UNK] for w in sen]


def LevenshteinDistance(s, t):
	if s == t:
		return 0
	if len(s) == 0:
		return len(t)
	if len(t) == 0:
		return len(s)
	v0 = [i for i in range(len(t)+1)]
	v1 = [0 for i in range(len(t)+1)]
	for i in range(len(s)):
		v1[0] = i + 1
		for j in range(len(t)):
			if (s[i] == t[j]):
				v1[j + 1] = v0[j]
			else:
				v1[j + 1] = min(v0[j], v0[j + 1], v1[j]) + 1
		for j in range(len(v0)):
			v0[j] = v1[j]
	return v1[len(t)]


def calculateWER(hyp, ref):
	"""
	Args:
		hyp(list of str): 候选译文, i.e. ['I', 'love', 'you']
		ref(list of str): 参考译文列表, i.e. ['I', 'like', 'you']
	Return:
		float: Word Error Rate
	"""
	return LevenshteinDistance(hyp, ref) * 1.0 / len(ref)


def index(ll, tgt, endToStart=False):
	begin, end, step = (len(ll)-1, -1, -1) if endToStart else (0, len(ll), 1)
	for i in range(begin, end, step):
		if ll[i] == tgt:
			return i
	raise ValueError()


def listToStr(ll, sep):
	return sep.join([str(item) for item in ll])


if __name__ == '__main__':
	ll = [5, 4, 3, 8, 4, 9]
	print(index(ll, 4), index(ll, 4, True))



