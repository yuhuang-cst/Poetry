# encoding: UTF-8

"""
@author: hy
"""

import tensorflow as tf
from Config import *
from NMTReader import *
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.bleu_score import corpus_bleu
import logging, logging.config
logging.config.fileConfig(CODE_PATH + '/logging.conf')
logger = logging.getLogger('consoleAndFile2')


class NMTSeq2DenseModel(object):
	def __init__(self, config, mode):
		self.name = 'NMTSeq2DenseModel'
		self.config = config
		self.mode = mode
		self.reader = NMTReader(self.config, NaiveBatchController)
		self.build()


	def build(self):
		self.embedding = tf.get_variable('embedding', shape=[self.config.vocabSize, self.config.embedSize], dtype=tf.float32)
		self.srcSeqIds = tf.placeholder(tf.int32, shape=[None, None], name='srcSeqId') # (batchSize, maxTime)
		self.seqLength = tf.placeholder(tf.int32, shape=[None], name='seqLength')    #batchSize
		self.batchSize, self.maxTime = tf.shape(self.srcSeqIds)[0], tf.shape(self.srcSeqIds)[1]
		srcSeqEmbed = tf.nn.embedding_lookup(self.embedding, self.srcSeqIds)   # (batchSize, maxTime)

		with tf.variable_scope('encoder'):
			fwEncoderCell = self.getMultiLSTMCell(self.config.fwLayers)
			bwEncoderCell = self.getMultiLSTMCell(self.config.bwLayers)
		# _(encoderOutput): (batchSize, maxTime, cell.outputSize);
		# encoderState: (numLayers, 2, batchSize, cell.stateSize), 2指lstm的(c, h)
		# _, self.encoderState = tf.nn.dynamic_rnn(
		# 	cell=encoderCell, inputs=srcSeqEmbed, sequence_length=self.srcSeqLength, dtype=tf.float32)

		# encoderOutput: (2, batchSize, maxTime, cell.outputSize)
		encoderOutput, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw=fwEncoderCell,
			cell_bw=bwEncoderCell,
			inputs=srcSeqEmbed,
			sequence_length=self.seqLength,
			dtype=tf.float32
		)
		encoderOutput = tf.concat(encoderOutput, axis=2)  # (batchSize, maxTime, cell.outputSize*2)
		fcOutput = tf.reshape(encoderOutput, shape=[-1, self.config.hiddenSize*2])

		for i in range(len(self.config.fcHiddenUnit)):
			fcOutput = tf.layers.dense(
				inputs=fcOutput, units=self.config.fcHiddenUnit[i], name='fc'+str(i),
				kernel_initializer=tf.get_variable_scope().initializer,
				bias_initializer=tf.get_variable_scope().initializer,
				reuse=tf.get_variable_scope().reuse
			)
		self.logits = tf.layers.dense(
			inputs=fcOutput, units=self.config.vocabSize, name='fc2',
			kernel_initializer=tf.get_variable_scope().initializer,
			bias_initializer=tf.get_variable_scope().initializer,
			reuse=tf.get_variable_scope().reuse
		)   # (batchSize, maxTime, vocabSize)
		self.logits = tf.reshape(self.logits, shape=[self.batchSize, self.maxTime, self.config.vocabSize])

		# INFER: output target id
		if self.mode == INFER_MODE:
			self.inferSeqIds = tf.argmax(self.logits, axis=2)   # (batchSize, maxTime)
			return

		# TRAIN or VALIDATE: calculate loss
		self.tgtSeqIds = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgtSeqIds')   # (batchSize, maxTime) [id(w1), ...,id(wn), id(eos)]
		self.loss = tf.contrib.seq2seq.sequence_loss(    # include softmax op
			logits=self.logits,
			targets=self.tgtSeqIds,
			weights= tf.sequence_mask(self.seqLength, dtype=tf.float32),
			average_across_timesteps=True,
			average_across_batch=True
		)

		# TRAIN: backpropagate
		if self.mode == TRAIN_MODE:
			# optimizer = tf.train.AdamOptimizer(self.config.lr)
			self.lr = tf.placeholder(dtype=tf.float32, name='LearningRate')
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
			self.optimizerName = optimizer.get_name()
			gradsAndVars = optimizer.compute_gradients(self.loss)
			gradients, variables = zip(*gradsAndVars)   #unzip
			gradients, _ = tf.clip_by_global_norm(gradients, self.config.maxGradNorm)
			gradsAndVars = zip(gradients, variables)
			self.globalStep = tf.train.get_or_create_global_step()   # create global_step node, used in Supervisord
			self.trainOp = optimizer.apply_gradients(gradsAndVars, global_step=self.globalStep)


	def getLSTMCell(self):
		lstmCell = tf.contrib.rnn.LSTMCell(
			num_units=self.config.hiddenSize,
			initializer=tf.get_variable_scope().initializer,
			reuse=tf.get_variable_scope().reuse
		)
		if self.mode == TRAIN_MODE:
			return tf.contrib.rnn.DropoutWrapper(lstmCell, output_keep_prob=self.config.keepRate)
		return lstmCell


	def getMultiLSTMCell(self, numLayers):
		return tf.contrib.rnn.MultiRNNCell([self.getLSTMCell() for _ in range(numLayers)])


	def train(self, sess, batchController, lr):
		assert self.mode == TRAIN_MODE
		for i in range(1, self.config.epochNum+1):
			srcSeqIds, tgtSeqIds, seqLength, _ = batchController.nextBatch(self.config.batchSize)
			_, self.globalStepNum, loss = sess.run([self.trainOp, self.globalStep, self.loss], feed_dict={
				self.srcSeqIds: srcSeqIds,
				self.seqLength: seqLength,
				self.tgtSeqIds: tgtSeqIds,
				self.lr: lr
			})
			if i % 100 == 0:
				logger.info('%d epoch, global step=%d, train loss=%f' % (i, self.globalStepNum, loss))


	def eval(self, sess, batchController):
		assert self.mode == TRAIN_MODE or self.mode == VALID_MODE
		lossTotal = 0.0
		count = 0
		notFetchNum = batchController.dataSize
		batchSize = self.config.batchSize
		while notFetchNum > 0:
			if notFetchNum < self.config.batchSize:
				batchSize = notFetchNum
			srcSeqIds, tgtSeqIds, seqLength, notFetchNum = batchController.nextBatch(batchSize)
			# print srcSeqLength.shape, type(srcSeqLength[0])
			lossTotal += sess.run(self.loss, feed_dict={
				self.srcSeqIds: srcSeqIds,
				self.seqLength: seqLength,
				self.tgtSeqIds: tgtSeqIds
			})
			count += 1
		return lossTotal / count


	def infer(self, sess, srcSeqIds, seqLength):
		assert self.mode == INFER_MODE
		inferSeqIds = sess.run(self.inferSeqIds, feed_dict={
			self.srcSeqIds: srcSeqIds,
			self.seqLength: seqLength
		})  # inferSeqIds.shape=(testNum, maxTime); inferSeqLength.shape=(testNum)
		TSSList = [self.reader.idsToSen(inferSeqIds[i][:seqLength[i]]) for i in range(len(seqLength))]
		return TSSList


	def inferSingle(self, sess, FS):
		assert self.mode == INFER_MODE
		srcSeqIds = [self.reader.senToIds(FS)]
		srcSeqLength = [len(srcSeqIds[0])]
		TSSList = self.infer(sess, srcSeqIds, srcSeqLength)
		return TSSList[0]


	def inferSingleAndPrint(self, sess, FS, SS):
		TSS = self.inferSingle(sess, FS)
		logger.info('%s %s |Ref:%s' % (FS, TSS, SS))


	def inferAll(self, sess, batchController, global_step):
		assert self.mode == INFER_MODE
		notFetchNum = batchController.dataSize
		batchSize = self.config.batchSize
		TSSList = []
		while notFetchNum > 0:
			if notFetchNum < self.config.batchSize:
				batchSize = notFetchNum
			srcSeqIds, _, seqLength, notFetchNum = batchController.nextBatch(batchSize)
			TSSList.extend(self.infer(sess, srcSeqIds, seqLength))
		pairList = batchController.allDataRaw()
		filepath = '%s/%s_gs%d_%s.txt' % (NMT_RESULT_PATH, self.name, global_step, self.config.toStr())
		saveResult(pairList, TSSList, filepath)
		refList = [[SS] for FS, SS in pairList]
		bleu = corpus_bleu(refList, TSSList, weights=SCORE_WEIGHT)
		gleu = corpus_gleu(refList, TSSList, min_len=1, max_len=len(SCORE_WEIGHT))
		infoStr = 'BLEU=%.4f, GLEU=%.4f |%s_%s' % (bleu, gleu, self.name, self.config.toStr())
		return bleu, gleu, infoStr


if __name__ == '__main__':
	from datetime import *
	startTime = datetime.now()

	config = NMTSeq2DenseConfig()
	initializer = tf.random_uniform_initializer(minval=-config.initScale, maxval=config.initScale)
	with tf.name_scope('Train'):
		with tf.variable_scope('Model', initializer=initializer):
			trainModel = NMTSeq2DenseModel(config, TRAIN_MODE)

	with tf.name_scope('Valid'):
		with tf.variable_scope('Model', initializer=initializer, reuse=True):
			validModel = NMTSeq2DenseModel(config, VALID_MODE)

	with tf.name_scope('Infer'):
		with tf.variable_scope('Model', initializer=initializer, reuse=True):
			inferModel = NMTSeq2DenseModel(config, INFER_MODE)

	initOp = tf.global_variables_initializer()
	savePath = NMT_MODEL_PATH+os.sep+trainModel.name+os.sep+trainModel.optimizerName+os.sep+config.toStr()
	# Adam
	# sv = tf.train.Supervisor(logdir=savePath, init_op=initOp, save_model_secs=1800, summary_op=None)
	# with sv.managed_session() as sess:
	# 	for i in range(1, config.hyEpochNum+1):
	# 		trainModel.train(sess, trainModel.reader.train)
	# 		validLoss = validModel.eval(sess, validModel.reader.valid)
	# 		logger.info('HyEpoch %d, valid loss=%f' % (i, validLoss))
	# 		if i % 5 == 0:
	# 			inferModel.inferSingleAndPrint(sess, '胡雁哀鸣夜夜飞', '胡儿眼泪双双落')    # test
	# 			inferModel.inferSingleAndPrint(sess, '新秋日后晒书天', '白日当松影却圆')    # test
	# 			inferModel.inferSingleAndPrint(sess, '佳人忆山水', '置酒在高台')            # train
	# 			inferModel.inferSingleAndPrint(sess, '朝朝奉御临池上', '不羡青松拜大夫')    # train
	# 		if i % 20 == 0:
	# 			bleu, gleu, infoStr = inferModel.inferAll(sess, inferModel.reader.test, sess.run(sv.global_step))
	# 			logger.info(infoStr)

	# SGD
	sv = tf.train.Supervisor(logdir=savePath, init_op=initOp, save_model_secs=600, summary_op=None)
	lr = config.lr
	with sv.managed_session() as sess:
		lr = 0.5
		logger.info('Learning Rate=%f', lr)
		for i in range(1, config.hyEpochNum+1):
			trainModel.train(sess, trainModel.reader.train, lr)
			validLoss = validModel.eval(sess, validModel.reader.valid)
			logger.info('HyEpoch %d, valid loss=%f' % (i, validLoss))
			if i % 5 == 0:
				inferModel.inferSingleAndPrint(sess, '胡雁哀鸣夜夜飞', '胡儿眼泪双双落')    # test
				inferModel.inferSingleAndPrint(sess, '新秋日后晒书天', '白日当松影却圆')    # test
				inferModel.inferSingleAndPrint(sess, '佳人忆山水', '置酒在高台')            # train
				inferModel.inferSingleAndPrint(sess, '朝朝奉御临池上', '不羡青松拜大夫')    # train
			if i % 20 == 0:
				bleu, gleu, infoStr = inferModel.inferAll(sess, inferModel.reader.test, sess.run(sv.global_step))
				logger.info(infoStr)
			# if i % 100 == 0:
			# 	lr /= 2
			# 	logger.info('Learning Rate=%f', lr)

	endTime = datetime.now()
	logger.info('Using Time =%s' % (str(endTime - startTime)))
