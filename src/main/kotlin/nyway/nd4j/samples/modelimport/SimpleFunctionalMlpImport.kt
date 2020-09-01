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
package nyway.nd4j.samples.modelimport

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.LossLayer
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File

/**
 * Basic example for importing a Keras functional Model into DL4J for training or inference.
 *
 * Let's say you want to create a simple MLP in Keras using the functional API. The model
 * takes mini-batches of vectors of length 100, has two Dense layers and predicts a total
 * of 10 categories. Such a model can be defined and serialized in HDF5 format as follows:
 *
 *
 * ```
 * from keras.models import Model
 * from keras.layers import Dense, Input
 *
 * inputs = Input(shape=(100,))
 * x = Dense(64, activation='relu')(inputs)
 * predictions = Dense(10, activation='softmax')(x)
 * model = Model(inputs=inputs, outputs=predictions)
 * model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])
 *
 * model.save('simple_functional_mlp.h5')
 * ```
 *
 * This model hasn't been fit on any data yet, it's just the model definition with initial weights
 * and training configuration as provided in the compile step.
 *
 * You don't need to create this model yourself to run this example, we stored it in the resources
 * folder under 'modelimport/keras' for you.
 *
 * This example shows you how to load the saved Keras model into DL4J for further use. You could
 * either use the imported model for inference only, or train it on data in DL4J.
 *
 *
 * @author Max Pumperla
 */
object SimpleFunctionalMlpImport {
    var dataLocalPath: String? = null

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        println("plip")
        dataLocalPath = "data/"
        val SIMPLE_FUNCTIONAL_MLP = File(dataLocalPath, "keras/simple_functional_mlp.h5").absolutePath

        // Keras functional Models correspond to DL4J ComputationGraphs. We enforce loading the training configuration
        // of the model as well. If you're only interested in inference, you can safely set this to 'false'.
        val model = KerasModelImport.importKerasModelAndWeights(SIMPLE_FUNCTIONAL_MLP, true)

        // Test basic inference on the model. Computation graphs take arrays of inputs and outputs, in this case of
        // length one.
        val input = arrayOf(Nd4j.create(256, 100))
        val output = model.output(*input)

        // Test basic model training.
        model.fit(input, output)
        assert(model.conf().optimizationAlgo == OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)

        // The first layer is a dense layer with 100 input and 64 output units, with RELU activation
        val first = model.getLayer(0)
        val firstConf = first.conf().layer as DenseLayer
        assert(firstConf.activationFn == Activation.RELU.activationFunction)
        assert(firstConf.nIn == 100L)
        assert(firstConf.nOut == 64L)

        // The second later is a dense layer with 64 input and 10 output units, with Softmax activation.
        val second = model.getLayer(1)
        val secondConf = second.conf().layer as DenseLayer
        assert(secondConf.activationFn == Activation.SOFTMAX.activationFunction)
        assert(secondConf.nIn == 64L)
        assert(secondConf.nOut == 10L)

        // The loss function of the Keras model gets translated into a DL4J LossLayer, which is the final
        // layer in this MLP.
        val loss = model.getLayer(2)
        val lossConf = loss.conf().layer as LossLayer
        assert(lossConf.lossFn == LossFunctions.LossFunction.MCXENT)
    }
}