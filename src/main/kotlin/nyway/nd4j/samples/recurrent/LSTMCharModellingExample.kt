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
package nyway.nd4j.samples.recurrent

import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.io.IOException
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.*

/**LSTM Character modelling example
 * @author Alex Black
 *
 * Example: Train a LSTM RNN to generates text, one character at a time.
 * This example is somewhat inspired by Andrej Karpathy's blog post,
 * "The Unreasonable Effectiveness of Recurrent Neural Networks"
 * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 *
 * This example is set up to train on the Complete Works of William Shakespeare, downloaded
 * from Project Gutenberg. Training on other text sources should be relatively easy to implement.
 *
 * For more details on RNNs in DL4J, see the following:
 * http://deeplearning4j.org/usingrnns
 * http://deeplearning4j.org/lstm
 * http://deeplearning4j.org/recurrentnetwork
 */
object LSTMCharModellingExample {
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val lstmLayerSize = 200 //Number of units in each LSTM layer
        val miniBatchSize = 32 //Size of mini batch to use when  training
        val exampleLength = 1000 //Length of each training example sequence to use. This could certainly be increased
        val tbpttLength = 50 //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        val numEpochs = 1 //Total number of training epochs
        val generateSamplesEveryNMinibatches = 10 //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        val nSamplesToGenerate = 4 //Number of samples to generate after each training epoch
        val nCharactersToSample = 300 //Length of each sample to generate
        val generationInitialization: String? = null //Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        val rng = Random(12345)

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our LSTM network.
        val iter = getShakespeareIterator(miniBatchSize, exampleLength)
        val nOut = iter.totalOutcomes()

        //Set up network configuration:
        val conf = NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam(0.005))
                .list()
                .layer(LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX) //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .build()
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(1))

        //Print the  number of parameters in the network (and for each layer)
        println(net.summary())

        //Do training, and then generate and print samples from network
        var miniBatchNumber = 0
        for (i in 0 until numEpochs) {
            while (iter.hasNext()) {
                val ds = iter.next()
                net.fit(ds)
                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    println("--------------------")
                    println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
                    println("Sampling characters from network given initialization \"" + (generationInitialization
                            ?: "") + "\"")
                    val samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
                    for (j in samples.indices) {
                        println("----- Sample $j -----")
                        println(samples[j])
                        println()
                    }
                }
            }
            iter.reset() //Reset iterator for another epoch
        }
        println("\n\nExample complete")
    }

    /** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
     * DataSetIterator that does vectorization based on the text.
     * @param miniBatchSize Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    @Throws(Exception::class)
    fun getShakespeareIterator(miniBatchSize: Int, sequenceLength: Int): CharacterIterator {
        //The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 million characters
        //https://www.gutenberg.org/ebooks/100
        val url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
        val tempDir = System.getProperty("java.io.tmpdir")
        val fileLocation = "$tempDir/Shakespeare.txt" //Storage location from downloaded file
        val f = File(fileLocation)
        if (!f.exists()) {
            FileUtils.copyURLToFile(URL(url), f)
            println("File downloaded to " + f.absolutePath)
        } else {
            println("Using existing text file at " + f.absolutePath)
        }
        if (!f.exists()) throw IOException("File does not exist: $fileLocation") //Download problem?
        val validCharacters: CharArray = CharacterIterator.minimalCharacterSet //Which characters are allowed? Others will be removed
        return CharacterIterator(fileLocation, StandardCharsets.UTF_8,
                miniBatchSize, sequenceLength, validCharacters, Random(12345))
    }

    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br></br>
     * Note that the initalization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more LSTM/RNN layers and a softmax output layer
     * @param iter CharacterIterator. Used for going from indexes back to characters
     */
    private fun sampleCharactersFromNetwork(initialization: String?, net: MultiLayerNetwork,
                                            iter: CharacterIterator, rng: Random, charactersToSample: Int, numSamples: Int): Array<String?> {
        //Set up initialization. If no initialization: use a random character
        var initialization = initialization
        if (initialization == null) {
            initialization = iter.randomCharacter.toString()
        }

        //Create input for initialization
        val initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length)
        val init = initialization.toCharArray()
        for (i in init.indices) {
            val idx = iter.convertCharacterToIndex(init[i])
            for (j in 0 until numSamples) {
                initializationInput.putScalar(intArrayOf(j, idx, i), 1.0f)
            }
        }
        val sb = arrayOfNulls<StringBuilder>(numSamples)
        for (i in 0 until numSamples) sb[i] = StringBuilder(initialization)

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState()
        var output = net.rnnTimeStep(initializationInput)
        output = output.tensorAlongDimension(output.size(2).toInt() - 1.toLong(), 1, 0) //Gets the last time step output
        for (i in 0 until charactersToSample) {
            //Set up next input (single time step) by sampling from previous output
            val nextInput = Nd4j.zeros(numSamples, iter.inputColumns())
            val cumsum = Nd4j.cumsum(output, 1)
            for (s in 0 until numSamples) {
                //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
                val sampledCharacterIdx = BooleanIndexing.firstIndex(cumsum.getRow(s.toLong()), Conditions.greaterThan(rng.nextDouble())).getInt(0)
                nextInput.putScalar(intArrayOf(s, sampledCharacterIdx), 1.0f) //Prepare next time step input
                sb[s]!!.append(iter.convertIndexToCharacter(sampledCharacterIdx)) //Add sampled character to StringBuilder (human readable output)
            }
            output = net.rnnTimeStep(nextInput) //Do one time step of forward pass
        }
        val out = arrayOfNulls<String>(numSamples)
        for (i in 0 until numSamples) out[i] = sb[i].toString()
        return out
    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    fun sampleFromDistribution(distribution: DoubleArray, rng: Random): Int {
        var d = 0.0
        var sum = 0.0
        for (t in 0..9) {
            d = rng.nextDouble()
            sum = 0.0
            for (i in distribution.indices) {
                sum += distribution[i]
                if (d <= sum) return i
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        throw IllegalArgumentException("Distribution is invalid? d=$d, sum=$sum")
    }
}