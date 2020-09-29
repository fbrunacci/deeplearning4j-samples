/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package nyway.nd4j.samples.tmp

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

/**
 * This is modified version of original LenetMnistExample, made compatible with multi-gpu environment and works using gradients sharing
 *
 * @author  @agibsonccc
 * @author raver119@gmail.com
 */
object GradientsSharingLenetMnistExample {
    private val log = LoggerFactory.getLogger(GradientsSharingLenetMnistExample::class.java)
    var nChannels = 1
    var outputNum = 10

    // for GPU you usually want to have higher batchSize
    var batchSize = 128
    var nEpochs = 2
    var seed = 123
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        log.info("Load data....")
        val mnistTrain: DataSetIterator = MnistDataSetIterator(batchSize, true, 12345)
        val mnistTest: DataSetIterator = MnistDataSetIterator(batchSize, false, 12345)
        log.info("Build model....")
        val conf = NeuralNetConfiguration.Builder()
                .seed(seed.toLong())
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(Nesterovs.Builder().learningRate(.01).build())
                .biasUpdater(Nesterovs.Builder().learningRate(0.02).build())
                .list()
                .layer(ConvolutionLayer.Builder(5, 5) //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(ConvolutionLayer.Builder(5, 5) //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
                .build()
        val model = MultiLayerNetwork(conf)
        model.init()

        model.setListeners(ScoreIterationListener(100))
        val timeX = System.currentTimeMillis()

        // optionally you might want to use MultipleEpochsIterator instead of manually iterating/resetting over your iterator
        //MultipleEpochsIterator mnistMultiEpochIterator = new MultipleEpochsIterator(nEpochs, mnistTrain);
        for (i in 0 until nEpochs) {
            val time1 = System.currentTimeMillis()
            model.fit(mnistTrain)
            val time2 = System.currentTimeMillis()
            log.info("*** Completed epoch {}, time: {} ***", i, time2 - time1)
        }
        val timeY = System.currentTimeMillis()
        log.info("*** Training complete, time: {} ***", timeY - timeX)
        log.info("Evaluate model....")
        val eval = model.evaluate<Evaluation>(mnistTest)
        log.info(eval.stats())
        log.info("****************Example finished********************")
    }
}