package nyway.nd4j.samples.mnist

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.activations.Activation

object MnistSample {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        //number of rows and columns in the input pictures
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val batchSize = 64 // batch size for each epoch
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 5 // number of epochs to perform
        val rate = 0.0015 // learning rate

        //Get the DataSetIterators:
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)


        val conf = NeuralNetConfiguration.Builder()
                .seed(rngSeed.toLong()) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm

                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Nesterovs(rate, 0.98)) //specify the rate of change of the learning rate.
                .l2(rate * 0.005) // regularize learning model
                .list()
                .layer(DenseLayer.Builder() //create the first input layer.
                        .nIn(numRows * numColumns)
                        .nOut(500)
                        .build())
                .layer(DenseLayer.Builder() //create the second input layer
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(outputNum)
                        .build())
                .build()

        val model = MultiLayerNetwork(conf)
        model.init()
        println(model.summary())

        model.setListeners(ScoreIterationListener(100))
        model.fit(mnistTrain, numEpochs)

        val eval: org.nd4j.evaluation.classification.Evaluation = model.evaluate(mnistTest)
        println(eval.stats())
    }

}