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
package nyway.nd4j.samples.sentiment

import nyway.nd4k.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.exp
import java.io.File
import kotlin.collections.set
import kotlin.math.*
import nyway.nd4k.*

object SentimentNetworkExample {

    @JvmStatic
    fun main(args: Array<String>) {

        Nd4j.getRandom().seed = 1234.toLong()
        val classLoader = this::class.java.classLoader
        val reviews = File(classLoader.getResource("./sentiment/reviews.txt").path).readLines()
        val labels = File(classLoader.getResource("./sentiment/labels.txt").path).readLines()

        val useTransform = true

        val vocab = mutableSetOf<String>()
        for (review in reviews) {
            for (word in review.split(" ")) {
                vocab.add(word)
            }
        }
        println("vocab.size=" + vocab.size)

        var sigmoid = { x: INDArray -> 1.0 / (1.0 + exp(-x)) }
        var sigmoid_derivative = { x: INDArray ->
            val sigmoidValue =  sigmoid(x)
            ( 1 - sigmoidValue ) * sigmoidValue
        }

        class SentimentNetwork(val vocab: Set<String>,
                               val activationFunction: (INDArray) -> INDArray,
                               val derivativeFunction: (INDArray) -> INDArray,
                               val hidden_nodes: Int = 10,
                               val output_nodes: Int = 1,
                               val learning_rate: Double = 0.1,
                               val printIterations: Int = 2500) {

            val input_nodes = this.vocab.size.toInt()

            var weights_0_1 = Nd4j.zeros(input_nodes, hidden_nodes)
            var weights_1_2 = Nd4j.randn(hidden_nodes.toLong(), output_nodes.toLong())

            val word2index = mutableMapOf<String, Int>()
            val label2index = mutableMapOf<String, Int>()

            init {
                preProcessData()
            }

            fun preProcessData() {
                for ((i, word) in vocab.withIndex())
                    word2index[word] = i

                for ((i, word) in vocab.withIndex())
                    label2index[word] = i
            }
        }

        fun SentimentNetwork.getInputLayer(review: String): INDArray {
            val arrayOfZeros = FloatArray(input_nodes.toInt())
            for (word in review.split(" ")) {
                if (word in word2index.keys) {
                    arrayOfZeros[word2index[word]!!] = 1f
                }
            }
            return Nd4j.create(arrayOfZeros, intArrayOf(1, input_nodes.toInt()))
        }

        fun SentimentNetwork.getTargetForLabel(label: String): Double {
            return when (label) {
                "positive" -> 1.0
                else -> 0.0
            }
        }

        fun SentimentNetwork.round(dbl: Number, decimals: Int = 2): Double {
            var multiplier = 10.0.pow(decimals)
            return Math.round(dbl.toDouble() * multiplier) / multiplier
        }

        fun SentimentNetwork.printDebugInfo(iteration: Int,
                                            reviewed: Int,
                                            correct_so_far: Int,
                                            printIterations: Int = 2500) {
            if (iteration % printIterations == 0 || iteration == reviewed) {
                val progress = round(100 * iteration / (reviewed))
                val training_accuracy = round(correct_so_far * 100 / iteration.toDouble())

                println("Progress:" + progress.toString().padStart(6) + "%"
                        + " #Correct:" + correct_so_far.toString().padStart(6)
                        + " #Iteration:" + iteration.toString().padStart(6)
                        + " Training Accuracy:" + training_accuracy.toString().padStart(4) + "%")
            }
        }

        fun SentimentNetwork.isPredictionCorrect(output: Float, label: String): Boolean {
            if ( (output >= 0.5 && label == "positive")
              || (output < 0.5 && label == "negative"))
                return true

            return false
        }

        fun SentimentNetwork.train(training_reviews: List<String>, training_labels: List<String>) {
            var correct_so_far = 0

            for ((i, review) in training_reviews.withIndex()) {
                val layer_0 = getInputLayer(review)

                // forward
                val layer_1 = layer_0.dot(weights_0_1)
                val layer_2 = activationFunction(layer_1.dot(weights_1_2))

                // back propagation
                val label = training_labels[i]
                val layer_2_error = layer_2 - getTargetForLabel(label)
                val layer_2_delta = layer_2_error * derivativeFunction(layer_2)

                val layer_1_error = layer_2_delta.dot(weights_1_2.T())
                val layer_1_delta = layer_1_error

                weights_1_2 -= layer_1.T().dot(layer_2_delta) * learning_rate
                weights_0_1 -= layer_0.T().dot(layer_1_delta) * learning_rate

                val output = layer_2.getFloat(0)
                if (isPredictionCorrect(output, label)) { correct_so_far++ }
                printDebugInfo(i + 1 , training_reviews.size, correct_so_far, printIterations)
            }
        }

        val training_reviews = reviews.take(2500 - 1000)
        val training_labels = labels.take(2500 - 1000)
        val test_reviews = reviews.takeLast(1000)
        val test_labels = labels.takeLast(1000)
        val mlp = SentimentNetwork(vocab, sigmoid, sigmoid_derivative, learning_rate=0.1)
        mlp.train(training_reviews,training_labels)

        fun SentimentNetwork.run(review: String): String {
            val layer_0 = getInputLayer(review)

            // forward
            val layer_1 = layer_0.dot(weights_0_1)
            val layer_2 = activationFunction(layer_1.dot(weights_1_2))
            val output = layer_2.getFloat(0)

            return if (output >= 0.5)
                "positive"
            else
                "negative"
        }

        fun SentimentNetwork.test(test_reviews: List<String>, test_labels: List<String>) {
            var correct = 0
            for ((i, review) in test_reviews.withIndex()) {
                val label = test_labels[i]
                val pred = run(review)
                if(pred ==label){
                    correct += 1
                }
                printDebugInfo(i + 1 , test_reviews.size, correct, printIterations)
            }
        }

        mlp.test(test_reviews,test_labels)

    }

}



