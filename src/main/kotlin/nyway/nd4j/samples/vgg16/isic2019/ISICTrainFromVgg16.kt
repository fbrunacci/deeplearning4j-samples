/* *****************************************************************************
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
 ******************************************************************************/
package nyway.nd4j.samples.vgg16.isic2019

import krangl.writeCSV
import nyway.nd4j.samples.Samples
import nyway.nd4j.samples.listeners.PrintScoreIterationListener
import org.deeplearning4j.core.storage.StatsStorage
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.zoo.ZooModel
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.io.IOException
import java.util.*


object ISICTrainFromVgg16 {
//    private val log = LoggerFactory.getLogger(EditLastLayerOthersFrozen::class.java)
    internal const val numClasses = 9
    internal const val seed: Long = 12345
    private const val batchSize = 1
    private const val featureExtractionLayer = "fc2"

    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val isicFolder = "${Samples.dataFolder}/ISIC_2019"

        val isicDataSet = ISICDataSet(
                "$isicFolder/ISIC_2019_Training_Input",
                "$isicFolder/ISIC_2019_Training_GroundTruth.csv",
                trainSize = 0.9f)

        val rng = Random(123)
        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        val zooModel: ZooModel<*> = VGG16.builder().build()
        val vgg16 = zooModel.initPretrained() as ComputationGraph
        println(vgg16.summary())


        // reduce the learning rate as the number of training epochs increases
        // iteration #, learning rate
        val learningRateSchedule: MutableMap<Int, Double> = HashMap()
        learningRateSchedule[0] = 5e-5
        learningRateSchedule[1000] = 5e-6
        learningRateSchedule[2000] = 5e-7
        learningRateSchedule[3000] = 5e-8

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        val fineTuneConf = FineTuneConfiguration.Builder()
//                .updater(Nesterovs(5e-5))
                .updater(Adam(0.0001))
//                .updater(Nadam(5e-5))
//                .updater(Nesterovs( MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
//                .updater(Nadam( MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                .seed(seed)
                .build()

        //Construct a new model with the intended architecture and print summary
        val vgg16Transfer = TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        OutputLayer.Builder(LossFunctions.LossFunction.KL_DIVERGENCE)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(NormalDistribution(0.0, 0.2 * (2.0 / (4096 + numClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build()
        println(vgg16Transfer.summary())

        val uiServer = UIServer.getInstance()
        val statsStorage: StatsStorage = InMemoryStatsStorage()
        uiServer.attach(statsStorage)

        val score = PrintScoreIterationListener(1)
        vgg16Transfer.setListeners(StatsListener(statsStorage), score)

        val trainIter = isicDataSet.trainIterator()
        val testIter = isicDataSet.testIterator()
        testIter.reset()

        var eval: Evaluation

        println("Number of image to train: ${isicDataSet.nTrain}")
        println("Number of image for test: ${isicDataSet.nTest}")

        //Print score every 10 iterations and evaluate on test set every epoch
        vgg16Transfer.setListeners(
                ScoreIterationListener(100)
                //,EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        )
        for(epoch in 1..10) {
            vgg16Transfer.fit(trainIter)
        }

        println("> vgg16Transfer.evaluate(testIter)")
        testIter.reset()
        eval = vgg16Transfer.evaluate(testIter)
        println(eval.stats())
        val vgg16TransferFile = File("${Samples.modelFolder}/vgg16/Vgg16OnISIC.zip")
        println("> saving model to $vgg16TransferFile")
        vgg16Transfer.save(vgg16TransferFile, true)
        println("> saving test data to $vgg16TransferFile")
        isicDataSet.testDataFrame.writeCSV(File("${Samples.modelFolder}/vgg16/Vgg16OnISIC_testDataFrame.csv"))
    }
}