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
package nyway.nd4j.samples.isic.isic2020

import krangl.count
import krangl.writeCSV
import nyway.nd4j.samples.Samples
import nyway.nd4j.samples.isic.isic2020.deotte.ISICDeotteDataSet
import nyway.nd4j.samples.isic.isic2020.deotte.IsicDeotteData
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.zoo.ZooModel
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.io.IOException
import java.util.*

/*
========================================================================================================
VertexName (VertexType)           nIn,nOut     TotalParams   ParamsShape                  Vertex Inputs 
========================================================================================================
input_1 (InputVertex)             -,-          -             -                            -             
block1_conv1 (ConvolutionLayer)   3,64         1 792         W:{64,3,3,3}, b:{1,64}       [input_1]     
block1_conv2 (ConvolutionLayer)   64,64        36 928        W:{64,64,3,3}, b:{1,64}      [block1_conv1]
block1_pool (SubsamplingLayer)    -,-          0             -                            [block1_conv2]
block2_conv1 (ConvolutionLayer)   64,128       73 856        W:{128,64,3,3}, b:{1,128}    [block1_pool] 
block2_conv2 (ConvolutionLayer)   128,128      147 584       W:{128,128,3,3}, b:{1,128}   [block2_conv1]
block2_pool (SubsamplingLayer)    -,-          0             -                            [block2_conv2]
block3_conv1 (ConvolutionLayer)   128,256      295 168       W:{256,128,3,3}, b:{1,256}   [block2_pool] 
block3_conv2 (ConvolutionLayer)   256,256      590 080       W:{256,256,3,3}, b:{1,256}   [block3_conv1]
block3_conv3 (ConvolutionLayer)   256,256      590 080       W:{256,256,3,3}, b:{1,256}   [block3_conv2]
block3_pool (SubsamplingLayer)    -,-          0             -                            [block3_conv3]
block4_conv1 (ConvolutionLayer)   256,512      1 180 160     W:{512,256,3,3}, b:{1,512}   [block3_pool] 
block4_conv2 (ConvolutionLayer)   512,512      2 359 808     W:{512,512,3,3}, b:{1,512}   [block4_conv1]
block4_conv3 (ConvolutionLayer)   512,512      2 359 808     W:{512,512,3,3}, b:{1,512}   [block4_conv2]
block4_pool (SubsamplingLayer)    -,-          0             -                            [block4_conv3]
block5_conv1 (ConvolutionLayer)   512,512      2 359 808     W:{512,512,3,3}, b:{1,512}   [block4_pool] 
block5_conv2 (ConvolutionLayer)   512,512      2 359 808     W:{512,512,3,3}, b:{1,512}   [block5_conv1]
block5_conv3 (ConvolutionLayer)   512,512      2 359 808     W:{512,512,3,3}, b:{1,512}   [block5_conv2]
block5_pool (SubsamplingLayer)    -,-          0             -                            [block5_conv3]
flatten (PreprocessorVertex)      -,-          -             -                            [block5_pool] 
fc1 (DenseLayer)                  25088,4096   102 764 544   W:{25088,4096}, b:{1,4096}   [flatten]     
fc2 (DenseLayer)                  4096,4096    16 781 312    W:{4096,4096}, b:{1,4096}    [fc1]         
predictions (DenseLayer)          4096,1000    4 097 000     W:{4096,1000}, b:{1,1000}    [fc2]         
--------------------------------------------------------------------------------------------------------
            Total Parameters:  138 357 544
        Trainable Parameters:  138 357 544
           Frozen Parameters:  0
========================================================================================================

 */
object ISICTrainDataFrameSet {

    internal const val numClasses = 2
    internal const val seed: Long = 12345
    private const val batchSize = 1
    private const val featureExtractionLayer = "fc2"

    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {

        val TAG ="Deotte"

        val deotteFolder = "${Samples.dataFolder}/ISIC_2020/Deotte/"
        val deotteIsisData = IsicDeotteData.buildData(deotteFolder)
        val isicDataFrameSet: IDataFrameSet = ISICDeotteDataSet(deotteIsisData, preProcessor = VGG16ImagePreProcessor())

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

        val trainIter = isicDataFrameSet.trainIterator()
        val testIter = isicDataFrameSet.testIterator()
        testIter.reset()

        var eval: Evaluation

        println("Train : ${isicDataFrameSet.trainDataFrame.groupBy("benign_malignant").count()}")
        println("Test : ${isicDataFrameSet.testDataFrame.groupBy("benign_malignant").count()}")

        //Print score every 10 iterations and evaluate on test set every epoch
        vgg16Transfer.setListeners(
                ScoreIterationListener(1000)
                , EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        )
        for(epoch in 1..10) {
            vgg16Transfer.fit(trainIter)
        }

//        println("> vgg16Transfer.evaluate(testIter)")
//        testIter.reset()
//        eval = vgg16Transfer.evaluate(testIter)
//        println(eval.stats())
        val vgg16TransferFile = File("${Samples.modelFolder}/vgg16_on_${TAG}_model.zip")
        println("> saving model to $vgg16TransferFile")
        vgg16Transfer.save(vgg16TransferFile, true)
        println("> saving test data to $vgg16TransferFile")
        isicDataFrameSet.testDataFrame.writeCSV(File("${Samples.modelFolder}/vgg16_on_${TAG}_testDataFrame.csv"))
    }
}