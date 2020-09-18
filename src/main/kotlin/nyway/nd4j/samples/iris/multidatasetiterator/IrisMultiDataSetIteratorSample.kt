package nyway.nd4j.samples.iris.multidatasetiterator

import nyway.nd4j.samples.iris.IrisClassifier
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File


object IrisMultiDataSetIteratorSample {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val classLoader = this::class.java.classLoader
        val irisFile = File(classLoader.getResource("./iris.csv").path)
        val dataPetalAndSepalIn = IrisClassifier.fullMatrix

        val dataSet = IrisMultiDataSet(irisFile.path)
        val trainIt = dataSet.trainIterator()
        val testIt = dataSet.testIterator()

        val conf = NeuralNetConfiguration.Builder()
                .seed(IrisClassifier.seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(Nadam()) //specify the rate of change of the learning rate.
                .l2(1e-4)
                .graphBuilder()
                .addInputs("P_IN", "S_IN")
                .addLayer("P_L1", DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build(),
                        "P_IN")
                .addLayer("P_L2",
                        DenseLayer.Builder()
                                .nIn(3)
                                .nOut(3)
                                .build(),
                        "P_L1")
                .addLayer("S_L1", DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build(),
                        "S_IN")
                .addLayer("S_L2",
                        DenseLayer.Builder()
                                .nIn(3)
                                .nOut(3)
                                .build(),
                        "S_L1")
                .addVertex("merge", MergeVertex(), "P_L2", "S_L2")
                .addLayer("L3", DenseLayer.Builder()
                        .nIn(6)
                        .nOut(3)
                        .build(), "merge")
                .addLayer("out", OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(3)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build(),
                        "L3")
                .setOutputs("out")
                .build()

        val model = ComputationGraph(conf)
        model.init()
        println(model.summary())
        model.setListeners(ScoreIterationListener(100))

        for (epoch in 1..2000) {
            trainIt.reset()
            while (trainIt.hasNext()) {
                val trainData = trainIt.next()
                model.fit(trainData)
            }
        }

        val eval = model.evaluate<Evaluation>(testIt)
        println("Score ${eval.stats()}")
    }

}