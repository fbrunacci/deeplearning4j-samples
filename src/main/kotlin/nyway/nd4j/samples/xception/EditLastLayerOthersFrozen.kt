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
package nyway.nd4j.samples.xception

import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.zoo.ZooModel
import org.deeplearning4j.zoo.model.Xception
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import nyway.nd4j.samples.vgg16.flower.EditLastLayerOthersFrozen
import java.io.File
import java.io.IOException

object EditLastLayerOthersFrozen {
//    private val log = LoggerFactory.getLogger(EditLastLayerOthersFrozen::class.java)
    internal const val numClasses = 5
    internal const val seed: Long = 12345
    const val trainPerc = 80
    private const val batchSize = 15
    private const val featureExtractionLayer = "avg_pool"

    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        val zooModel: ZooModel<*> = Xception.builder().build()
        val xception = zooModel.initPretrained() as ComputationGraph
        println(xception.summary())

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        val fineTuneConf = FineTuneConfiguration.Builder()
                .updater(Nesterovs(5e-5))
                .seed(seed)
                .build()

        //Construct a new model with the intended architecture and print summary
        val xceptionTransfer = TransferLearning.GraphBuilder(xception)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(EditLastLayerOthersFrozen.numClasses)
                                .weightInit(NormalDistribution(0.0, 0.2 * (2.0 / (4096 + EditLastLayerOthersFrozen.numClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build()
        println(xception.summary())



    }
}