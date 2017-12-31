# encoding: UTF-8

"""
@author: hy
@run: python SMT.py -n <LM_GRAM_NUM> -s <SMOOTHING_LAMBDA> -w <LM_WEIGHT> -b <BEAM_SIZE>
"""

import sys, getopt
from Config import *
from collections import Counter
import numpy as np
from Common import *
import heapq
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.bleu_score import corpus_bleu
import logging, logging.config

logging.config.fileConfig(CODE_PATH + '/logging.conf')
logger = logging.getLogger('consoleAndFile')

class SMTModel(object):
	def __init__(self, config):
		self.LM_GRAM_NUM = config.lmGramNum
		self.SMOOTHING_LAMBDA = config.smLambda
		self.LM_WEIGHT = config.lmWeight
		self.BEAM_SIZE = config.beamSize
		self.data = None  # {'train': [(FS, SS), ...], 'lm': [sen, ]}
		self.lmGramDict = {}    # {(w1, ..., w_(n-1)): {wn: logp, NEVER_OCCUR_CHAR: logp, ..}, NO_CHAR: {}, ..}
		self.transGramDict = {} # {SW: {FW: logp, NEVER_OCCUR_CHAR: logp}}; SW: word of SS; FW: word of FS
		self.lmVCounter = set()
		self.trainVCounter = set()
		self.trainVList = []


	def generateVCounter(self):
		_, _, self.lmVCounter = countCharacter(self.data['lm'])
		_, _, self.trainVCounter = countCharacter([FS+SS for FS, SS in self.data['train']])
		self.trainVList = self.trainVCounter.keys()


	def calTransDict(self):
		logger.info('Traning Translation Model...')
		for FS, SS in self.data['train']:
			for i in range(len(SS)):
				if SS[i] in self.transGramDict:
					self.transGramDict[SS[i]][FS[i]] += 1
				else:
					self.transGramDict[SS[i]] = Counter(FS[i])
		for given in self.transGramDict:
			totalCount = sum(self.transGramDict[given].values())
			for w in self.transGramDict[given]:
				self.transGramDict[given][w] = \
					np.log(1.0 * (self.transGramDict[given][w] + self.SMOOTHING_LAMBDA) / (totalCount + len(self.lmVCounter) * self.SMOOTHING_LAMBDA))
			self.transGramDict[given][NEVER_OCCUR_CHAR] = \
				np.log(1.0 * self.SMOOTHING_LAMBDA / (totalCount + len(self.lmVCounter) * self.SMOOTHING_LAMBDA))
		self.transGramDict[NEVER_OCCUR_CHAR] = {NEVER_OCCUR_CHAR:  np.log(1.0 / len(self.trainVCounter))}
		# self.transGramDict[NEVER_OCCUR_CHAR] = {}
		# for w in self.trainVCounter:
		# 	self.transGramDict[NEVER_OCCUR_CHAR][w] = 1.0 / len(self.trainVCounter)
		# self.transGramDict[NEVER_OCCUR_CHAR][NEVER_OCCUR_CHAR] = 1.0 / len(self.trainVCounter)


	def calLMDict(self):
		logger.info('Training Language Model...')
		for n in range(1, self.LM_GRAM_NUM+1):
			for sen in self.data['lm']:
				for i in range(len(sen)-n+1):
					given = tuple(sen[i:i+n-1]) if n > 1 else (NO_CHAR,)
					if given in self.lmGramDict:
						self.lmGramDict[given][sen[i+n-1]] += 1
					else:
						self.lmGramDict[given] = Counter(sen[i+n-1])
		for given in self.lmGramDict:
			totalCount = sum(self.lmGramDict[given].values())
			for w in self.lmGramDict[given]:
				self.lmGramDict[given][w] = \
					np.log(1.0 * (self.lmGramDict[given][w] + self.SMOOTHING_LAMBDA) / (totalCount + len(self.lmVCounter) * self.SMOOTHING_LAMBDA))
			self.lmGramDict[given][NEVER_OCCUR_CHAR] = \
				np.log(1.0 * self.SMOOTHING_LAMBDA / (totalCount + len(self.lmVCounter) * self.SMOOTHING_LAMBDA))
		self.lmGramDict[(NEVER_OCCUR_CHAR,)] = {NEVER_OCCUR_CHAR: np.log(1.0 / len(self.lmVCounter))}


	def train(self, trainData, lmData):
		"""
		Args:
			trainData (list): [[FS, SS], ...], type(FS)=type(SS)=unicode
			lmData (list): [sen1, sen2, ...], type(sen)=unicode
		"""
		logger.info('Training...')
		self.data = {'train': trainData, 'lm': lmData}
		self.generateVCounter()
		self.calLMDict()
		self.calTransDict()


	def lmGivenWrap(self, given):
		noCharNum = given.count(NO_CHAR)
		if noCharNum == len(given):
			return (NO_CHAR,)
		return tuple(given[noCharNum:])


	def stripNoChar(self, wList):
		b, e = 0, len(wList)-1
		while b < len(wList):
			if wList[b] != NO_CHAR:
				break
			b += 1
		while e > -1:
			if wList[e] != NO_CHAR:
				break
			e -= 1
		return wList[b: e+1]


	def calTranslogp(self, sw, fw):
		"""
		Args:
			sw (unicode): word of second sentence
			fw (unicode): word of first sentence
		Returns:
			float: logp(fw|sw)
		"""
		sw = keyWrap(sw, self.transGramDict, NEVER_OCCUR_CHAR)
		fw = keyWrap(fw, self.transGramDict[sw], NEVER_OCCUR_CHAR)
		return self.transGramDict[sw][fw]


	def calLMlogp(self, given, w):
		"""
		Args:
			given (list): [w1, w2, ..., w_(n-1)]
			w (unicode): wn
		Returns:
			float: logp(w|given)
		"""
		given = keyWrap(self.lmGivenWrap(given), self.lmGramDict, (NEVER_OCCUR_CHAR,))
		w = keyWrap(w, self.lmGramDict[given], NEVER_OCCUR_CHAR)
		return self.lmGramDict[given][w]


	def generateSS(self, FS):
		"""
		Args:
			FS (unicode): first line
		Returns:
			list: [{'logp': logp, 'SS': SS}], 按logp降序
		"""
		# print 'FS =', FS
		beam = [{'logp':0.0, 'SS': [NO_CHAR for j in range(self.LM_GRAM_NUM-1)]}]
		for fw in FS:
			# print fw,
			# for i in range(len(beam)):
			# 	print beam[i]['logp'], ''.join(beam[i]['SS']), ';',
			# print ''
			logpList = []   # [(logp, index(in beam), word), ...]
			trlogpList = np.array(list(map(lambda sw: self.calTranslogp(sw, fw), self.trainVList)))   # shape=(transV,)
			for j in range(len(beam)):
				given = beam[j]['SS'][len(beam[j]['SS'])-self.LM_GRAM_NUM+1:]
				lmlogpList = np.array(list(map(lambda sw: self.calLMlogp(given, sw), self.trainVList)))
				logpJ = (1.0 - self.LM_WEIGHT)*trlogpList + self.LM_WEIGHT*lmlogpList + beam[j]['logp']
				logpList.extend(list(map(lambda logp, sw: (logp, j, sw), logpJ, self.trainVList)))

			logpList = heapq.nlargest(self.BEAM_SIZE, logpList)
			# for logp, index, sw in logpList[:20]:
			# 	print logp, index, sw
			newBeam = [{'logp': logp, 'SS': beam[index]['SS']+[sw]} for logp, index, sw in logpList]
			beam = newBeam
		for i in range(len(beam)):
			beam[i]['SS'] = ''.join(self.stripNoChar(beam[i]['SS']))
		return beam


	def generateSSList(self, FSList, pFlag=False):
		"""
		Args:
			FSList (list of unicode): [FS1, FS2, ...]
			pFlag (bool): print flag
		Returns:
			(list of unicode): TSSList, [TSS1, TSS2, ...], TSS=Target Seconde Sentence
		"""
		TSSList = []
		size = len(FSList)
		for i in range(size):
			TSSList.append(self.generateSS(FSList[i])[0]['SS'])
			if pFlag and i % 20 == 0:
				logger.info('%.2f%%(%d/%d) SS generated' % (100.0*(i+1)/size, i+1, size))
		return TSSList


	def generateAndPrint(self, FS, topk=None):
		if topk == None:
			topk = self.BEAM_SIZE
		beam = self.generateSS(FS)
		logger.info('-------------------------------\nFS =%s' % (FS))
		for SS in beam[:min(len(beam), topk)]:
			print('%s %f' % (SS['SS'], SS['logp']))


	def generageBest(self, FS):
		return self.generateSS(FS)[0]['SS']


	def test(self, testData):
		"""
		Args:
			testData (list): [[FS, SS], ...], type(FS)=type(SS)=unicode
		"""
		TSSList = self.generateSSList([FS for FS, SS in testData])
		filepath = SMT_RESULT_PATH+'/poemSMT_lmn%d_sm%.3f_lmw%.3f_be%d.txt' % (self.LM_GRAM_NUM, self.SMOOTHING_LAMBDA, self.LM_WEIGHT, self.BEAM_SIZE)
		saveResult(testData, TSSList, filepath)
		refList = [[SS] for FS, SS in testData]
		bleu = corpus_bleu(refList, TSSList, weights=SCORE_WEIGHT)
		gleu = corpus_gleu(refList, TSSList, min_len=1, max_len=len(SCORE_WEIGHT))
		infoStr = 'BLEU=%.4f, GLEU=%.4f |poemSMT_lmn%d_sm%.3f_lmw%.3f_be%d' % (bleu, gleu, self.LM_GRAM_NUM, self.SMOOTHING_LAMBDA, self.LM_WEIGHT, self.BEAM_SIZE)
		return bleu, gleu, infoStr



if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("-m", "--mode",  default = INTERACTIVE_MODE, help = "running mode: %s|%s; DEFAULT=%s" % (INTERACTIVE_MODE, TEST_MODE, INTERACTIVE_MODE), action = "store", type = "string", dest = "mode")
	(options, args) = parser.parse_args()
	data = readData(
		trainPath=DATA_PATH + '/preprocess/train.txt',
		validationPath=DATA_PATH + '/preprocess/validation.txt',
		testPath=DATA_PATH + '/preprocess/test_small.txt',
		lmPath=DATA_PATH + '/preprocess/LM.txt'
	)
	model = SMTModel(SMTConfig())
	model.train(data['train'], data['lm'])

	if options.mode == TEST_MODE:
		bleu, gleu, info = model.test(data['test'])
		logger.info(info)
	elif options.mode == INTERACTIVE_MODE:
		while True:
			FS = input('输入诗句: ')
			if FS == 'exit':
				exit()
			SS = model.generageBest(FS)
			print('输出诗句: %s' % (SS))


