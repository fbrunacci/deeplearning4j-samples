package nyway.nd4j.samples.iris

import krangl.DataFrame
import krangl.head
import krangl.readDelim
import krangl.shuffle
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
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.StringReader
import java.util.*

object IrisClassifierSample {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        println(IrisClassifier.data.head())

        val dataIn = Nd4j.create(IrisClassifier.fullMatrix)
        val dataOut = Nd4j.create(IrisClassifier.twodimLabel)
        val model = MultiLayerNetwork(IrisClassifier.neuralNetConf(dataIn.columns()))

        val fullDataSet = DataSet(dataIn, dataOut)
        fullDataSet.shuffle(IrisClassifier.seed)

        val splitedSet = fullDataSet.splitTestAndTrain(0.90)
        val trainingData = splitedSet.train;
        val testData = splitedSet.test;

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        val normalizer: DataNormalization = NormalizerStandardize()
        normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData) //Apply normalization to the training data
        normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set

        // train the network
        model.setListeners(ScoreIterationListener(100))
        for (l in 0..2000) {
            model.fit(trainingData)
        }
        // evaluate the network
        val eval = Evaluation()
        val output: INDArray = model.output(testData.features)
        eval.eval(testData.labels, output)
        println("Score " + eval.stats())
    }
}