package nyway.nd4j.samples.iris

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
import java.util.*

/**
 * Simple sample using vertex for merging multiple input
 */
object IrisMergeVertexSample {


    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        println("petal length/width")
        println(Arrays.deepToString(IrisClassifier.petalMatrix.take(5).toTypedArray()).replace("], ", "]\n")+"\n...")
        println("sepal length/width")
        println(Arrays.deepToString(IrisClassifier.sepalMatrix.take(5).toTypedArray()).replace("], ", "]\n")+"\n...")
        println("full data length/width")
        println(Arrays.deepToString(IrisClassifier.fullMatrix.take(5).toTypedArray()).replace("], ", "]\n")+"\n...")

        println("label to number")
        println(Arrays.deepToString(IrisClassifier.twodimLabel.take(5).toTypedArray()).replace("], ", "]\n")+"\n...")

        //Convert the data matrices into training INDArrays
        val dataPetalIn = Nd4j.create(IrisClassifier.petalMatrix)
        val dataSepalIn = Nd4j.create(IrisClassifier.sepalMatrix)
        val dataPetalAndSepalIn = Nd4j.create(IrisClassifier.fullMatrix)
        val dataOut = Nd4j.create(IrisClassifier.twodimLabel)

        val seed: Long = 6
        val petalOnlyEval = runNet("Petal", dataPetalIn)
        val sepalOnlyEval = runNet("Sepal", dataSepalIn)

        val mergedConf = NeuralNetConfiguration.Builder()
                .seed(seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(Nadam()) //specify the rate of change of the learning rate.
                .l2(1e-4)
                .graphBuilder()
                .addInputs("P_IN", "IN")
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
                .addLayer("L1", DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build(),
                        "IN")
                .addLayer("L2",
                        DenseLayer.Builder()
                                .nIn(3)
                                .nOut(3)
                                .build(),
                        "L1")
                .addVertex("merge", MergeVertex(), "P_L2", "L2")
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

        val mergedModel = ComputationGraph(mergedConf)
        mergedModel.init()
        println(mergedModel.summary())

        val fullDataSet = DataSet(dataPetalAndSepalIn, dataOut)
        fullDataSet.shuffle(seed)

        val splitedSet = fullDataSet.splitTestAndTrain(0.90)
        val trainingData = splitedSet.train;
        val testData = splitedSet.test;

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        val normalizer: DataNormalization = NormalizerStandardize()
        normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData) //Apply normalization to the training data
        normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set

        val trainingPetalDataSet = trainingData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2))
        val trainingSepalDataSet = trainingData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(2,4))

        val trainingDataSet = MultiDataSet(arrayOf(trainingPetalDataSet,trainingSepalDataSet),arrayOf(trainingData.labels))

        // train the network
        mergedModel.setListeners(ScoreIterationListener(100))
        for (l in 0..2000) {
            mergedModel.fit(trainingDataSet)
        }

        // evaluate the network
        val testPetalDataSet = testData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2))
        val testSepalDataSet = testData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(2,4))
        //val testDataSet = MultiDataSet(arrayOf(testPetalDataSet,testSepalDataSet),arrayOf(testData.labels))

        println("Petal Score " + petalOnlyEval.stats())
        println("Sepal Score " + sepalOnlyEval.stats())
        val mergedEval = Evaluation()
        val mergedOutput: INDArray = mergedModel.output(testPetalDataSet,testSepalDataSet).first()
        mergedEval.eval(testData.labels, mergedOutput)
        println("Merged Score " + mergedEval.stats())
    }

    private fun runNet(dataName: String, dataIn: INDArray): Evaluation {
        val conf = IrisClassifier.neuralNetConf(dataIn.columns())
        val model = MultiLayerNetwork(conf)

        //Create a data set from the INDArrays and shuffle it
        val dataSet = DataSet(dataIn, Nd4j.create(IrisClassifier.twodimLabel))
        dataSet.shuffle(IrisClassifier.seed)

        val splitedSet = dataSet.splitTestAndTrain(0.90)
        val trainingData = splitedSet.train;
        val testData = splitedSet.test;

        println("trainingData size: "+ trainingData.asList().size)
        println("testData size: "+ testData.asList().size)

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        val normalizer: DataNormalization = NormalizerStandardize()
        normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData) //Apply normalization to the training data
        normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set

        dataSet.features

        // train the network
        model.setListeners(ScoreIterationListener(100))
        for (l in 0..2000) {
            model.fit(trainingData)
        }

        // evaluate the network
        val eval = Evaluation()
        val output: INDArray = model.output(testData.features)
        eval.eval(testData.labels, output)
        println(model.summary())
        println(dataName + "Score " + eval.stats())
        return eval
    }

}