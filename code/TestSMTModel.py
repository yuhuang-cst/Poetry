# encoding: UTF-8

"""
@author: hy
"""

from SMT import SMTModel
from Config import *
from Common import *

class TestSMTModel(object):
	def __init__(self):
		data = readData(
			trainPath=DATA_PATH + '/preprocess/train.txt',
			validationPath=DATA_PATH + '/preprocess/validation.txt',
			testPath=DATA_PATH + '/preprocess/test.txt',
			lmPath=DATA_PATH + '/preprocess/LM.txt'
		)
		self.model = SMTModel(SMTConfig())
		self.model.train(data['train'], data['lm'])


	def testAll(self):
		print('test transGramDict:')
		self.testGramDictNeg(self.model.transGramDict)
		self.testTransGramRank()
		print('test lmGramDict:')
		self.testGramDictNeg(self.model.lmGramDict)
		self.testLMGramRank()


	def testGenerateAndPrint(self):
		self.model.generateAndPrint('听钟投宿入孤烟')
		self.model.generateAndPrint('佳人忆山水')
		self.model.generateAndPrint('朝朝奉御临池上')
		self.model.generateAndPrint('幽人听达曙')
		self.model.generateAndPrint('降集翻翔凤')
		self.model.generateAndPrint('圣主此时思共理')
		self.model.generateAndPrint('化城若化出')
		self.model.generateAndPrint('樱桃未绽梅花老')
		self.model.generateAndPrint('万里亲朋散')
		self.model.generateAndPrint('五马踟蹰在路岐')


	def testGramDictNeg(self, gramDict):
		for k1 in gramDict:
			for k2 in gramDict[k1]:
				if gramDict[k1][k2] > 0:
					print(k1, k2, gramDict[k1][k2])


	def testTransGramRank(self):
		# self.transGramDict = {} # {SW: {FW: logp, NEVER_OCCUR_CHAR: logp}}; SW: word of SS; FW: word of FS
		testSW = u'听钟投宿入孤烟佳人忆山水'
		for sw in testSW:
			print(u'--------------------------')
			for fw, logp in sorted(self.model.transGramDict[sw].items(), key=lambda item: item[1], reverse=True)[:20]:
				print(sw, fw, logp)


	def testLMGramRank(self):
		# self.lmGramDict = {}    # {(w1, ..., w_(n-1)): {wn: logp, NEVER_OCCUR_CHAR: logp, ..}, NO_CHAR: {}, ..}
		testSen = '听钟投宿入孤烟佳人忆山水'
		lmGramDict = self.model.lmGramDict
		print('--------------------------')
		for w in testSen:
			wrapW = keyWrap(w, lmGramDict[(NO_CHAR,)], (NEVER_OCCUR_CHAR,))
			print('', '->', (NO_CHAR,), wrapW, lmGramDict[(NO_CHAR,)][w])
		print('--------------------------')
		for w, logp in sorted(lmGramDict[(NO_CHAR,)].items(), key=lambda item: item[1], reverse=True)[:20]:
			print('', '->', (NO_CHAR,), w, logp)
		for n in range(2, self.model.LM_GRAM_NUM+1):
			for i in range(len(testSen)-n+1):
				print('--------------------------')
				given = self.model.lmGivenWrap(testSen[i:i+n-1])
				given = keyWrap(given, lmGramDict, (NEVER_OCCUR_CHAR,))
				for w, logp in sorted(lmGramDict[given].items(), key=lambda item: item[1], reverse=True)[:20]:
					print(testSen[i:i+n-1], u'->', given, w, logp)


if __name__ == '__main__':
	testModel = TestSMTModel()
	testModel.testGenerateAndPrint()
	# testModel.testAll()


