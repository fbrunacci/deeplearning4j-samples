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
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import java.io.File
import java.io.IOException
import java.util.*

/*
==================================================================================================================================================================================
VertexName (VertexType)                        nIn,nOut    TotalParams   ParamsShape                                                  Vertex Inputs
==================================================================================================================================================================================
input_1 (InputVertex)                          -,-         -             -                                                            -
stem_conv (ConvolutionLayer)                   3,56        1 512         W:{56,3,3,3}                                                 [input_1]
stem_bn (BatchNormalization)                   56,56       224           gamma:{1,56}, beta:{1,56}, mean:{1,56}, var:{1,56}           [stem_conv]
stem_activation (ActivationLayer)              -,-         0             -                                                            [stem_bn]
block1a_dwconv (DepthwiseConvolution2DLayer)   56,56       504           W:{3,3,56,1}                                                 [stem_activation]
block1a_bn (BatchNormalization)                56,56       224           gamma:{1,56}, beta:{1,56}, mean:{1,56}, var:{1,56}           [block1a_dwconv]
...
top_conv (ConvolutionLayer)                    576,2304    1 327 104     W:{2304,576,1,1}                                             [block7c_add]
top_bn (BatchNormalization)                    2304,2304   9 216         gamma:{1,2304}, beta:{1,2304}, mean:{1,2304}, var:{1,2304}   [top_conv]
top_activation (ActivationLayer)               -,-         0             -                                                            [top_bn]
avg_pool (GlobalPoolingLayer)                  -,-         0             -                                                            [top_activation]
top_dropout (DropoutLayer)                     -,-         0             -                                                            [avg_pool]
probs (DenseLayer)                             2304,1000   2 305 000     W:{2304,1000}, b:{1,1000}                                    [top_dropout]
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            Total Parameters:  43 265 136
        Trainable Parameters:  43 265 136
           Frozen Parameters:  0
==================================================================================================================================================================================

 */
object ISICTrainEfficientNetB6 {

    internal const val numClasses = 2
    internal const val seed: Long = 12345
    private const val batchSize = 1
    private const val featureExtractionLayer = "fc2"

    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {

        val TAG = "Deotte"
        val MODEL_NAME = "EfficientNet"

        val deotteFolder = "${Samples.dataFolder}/ISIC_2020/Deotte/"
        val deotteIsisData = IsicDeotteData.buildData(deotteFolder)
        val isicDataFrameSet: IDataFrameSet = ISICDeotteDataSet(deotteIsisData,
                height = 384, width = 384, nchw = false)

        val rng = Random(123)

        //InputType.setDefaultCNN2DFormat(CNN2DFormat.NCHW)
        val model = ModelSerializer.restoreComputationGraph("${Samples.modelFolder}/weka/KerasEfficientNetB6.zip")


        println(model.summary())

        //if( true ) return

        val trainIter = isicDataFrameSet.trainIterator()
        val testIter = isicDataFrameSet.testIterator()
        testIter.reset()

        var eval: Evaluation

        println("Train : ${isicDataFrameSet.trainDataFrame.groupBy("benign_malignant").count()}")
        println("Test : ${isicDataFrameSet.testDataFrame.groupBy("benign_malignant").count()}")

        //Print score every 10 iterations and evaluate on test set every epoch
        model.setListeners(
                ScoreIterationListener(100), EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        )
        for (epoch in 1..12) {
            model.fit(trainIter)
        }

        val transferFile = File("${Samples.modelFolder}/${MODEL_NAME}_on_${TAG}_model.zip")
        println("> saving model to $transferFile")
        model.save(transferFile, true)
        println("> saving test data to $transferFile")
        isicDataFrameSet.testDataFrame.writeCSV(File("${Samples.modelFolder}/${MODEL_NAME}_on_${TAG}_testDataFrame.csv"))
    }
}