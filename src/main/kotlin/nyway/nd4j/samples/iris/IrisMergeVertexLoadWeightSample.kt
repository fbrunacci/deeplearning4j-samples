package nyway.nd4j.samples.iris

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File

/**
 * https://stackoverflow.com/questions/42806761/initialize-custom-weights-in-deeplearning4j
 */
object IrisMergeVertexLoadWeightSample {


    private val iterations = 2000

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        //Convert the data matrices into training INDArrays
        val dataPetalIn = Nd4j.create(IrisClassifier.petalMatrix)
        val dataSepalIn = Nd4j.create(IrisClassifier.sepalMatrix)
        val dataPetalAndSepalIn = Nd4j.create(IrisClassifier.fullMatrix)
        val dataOut = Nd4j.create(IrisClassifier.twodimLabel)

        val petalConf = NeuralNetConfiguration.Builder()
                .seed(IrisClassifier.seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(Nadam()) //specify the rate of change of the learning rate.
                .l2(1e-4)
                .graphBuilder()
                .addInputs("P_IN")
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
                .addLayer("P_out", OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(3)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build(),
                        "P_L2")
                .setOutputs("P_out")
                .build()

        val sepalConf = NeuralNetConfiguration.Builder()
                .seed(IrisClassifier.seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(Nadam()) //specify the rate of change of the learning rate.
                .l2(1e-4)
                .graphBuilder()
                .addInputs("S_IN")
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
                .addLayer("S_out", OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(3)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build(),
                        "S_L2")
                .setOutputs("S_out")
                .build()

        //Create a data set from the INDArrays and shuffle it
        val splitedPetalDataSet = splitedDataSet(dataPetalIn)
        val splitedSepalDataSet = splitedDataSet(dataSepalIn)

        val useSavedModel = true
        val petalOnlyModel = runNet("Petal", petalConf, splitedPetalDataSet, useSavedModel)
        val sepalOnlyModel = runNet("Sepal", sepalConf, splitedSepalDataSet, useSavedModel)
//        println("petalOnlyModel:")
//        showParams(petalOnlyModel)
//        println("sepalOnlyModel:")
//        showParams(sepalOnlyModel)

//        showParams(petalOnlyModel)
//        val fineTuneConf = FineTuneConfiguration.Builder()
//                .seed(IrisClassifier.seed)
//                .build()
//        val petalFrozen = TransferLearning.GraphBuilder(petalOnlyModel)
//                .fineTuneConfiguration(fineTuneConf)
//                .setFeatureExtractor("P_L2")
//                .removeVertexKeepConnections("P_out")
//                .addLayer("P_out", OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nIn(3)
//                        .nOut(3)
//                        .activation(Activation.SOFTMAX)
//                        .weightInit(WeightInit.XAVIER)
//                        .build(),
//                        "P_L2")
//                .build()
//        println(petalFrozen.summary())
//        showParams(petalFrozen)

        val mergedConf = NeuralNetConfiguration.Builder()
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

        val mergedModel = ComputationGraph(mergedConf)
        mergedModel.init()

        println(mergedModel.summary())
        copyParams(petalOnlyModel, mergedModel)
        copyParams(sepalOnlyModel, mergedModel)

        println("mergedModel:")
        showParams(mergedModel)
        println(mergedModel.summary())

        val splitedSet = splitedDataSet(dataPetalAndSepalIn)
        val trainingData = splitedSet.train;
        val testData = splitedSet.test;

        val trainingPetalDataSet = trainingData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2))
        val trainingSepalDataSet = trainingData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 4))
        val trainingDataSet = MultiDataSet(arrayOf(trainingPetalDataSet, trainingSepalDataSet), arrayOf(trainingData.labels))

        // train the network
        mergedModel.setListeners(ScoreIterationListener(100))
        for (l in 0..iterations) {
            mergedModel.fit(trainingDataSet)
        }

        // evaluate the network
        val testPetalDataSet = testData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2))
        val testSepalDataSet = testData.features.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 4))
        //val testDataSet = MultiDataSet(arrayOf(testPetalDataSet,testSepalDataSet),arrayOf(testData.labels))

        val mergedEval = Evaluation()
        val mergedOutput: INDArray = mergedModel.output(testPetalDataSet, testSepalDataSet).first()
        mergedEval.eval(testData.labels, mergedOutput)
        println("Merged Score " + mergedEval.stats())


    }

    private fun splitedDataSet(dataPetalIn: INDArray, dataOut: INDArray = Nd4j.create(IrisClassifier.twodimLabel)): SplitTestAndTrain {
        val dataSet = DataSet(dataPetalIn, dataOut)
        dataSet.shuffle(IrisClassifier.seed)
        val normalizer: DataNormalization = NormalizerStandardize()

        val splitedDataSet = dataSet.splitTestAndTrain(0.90)
        normalizer.fit(splitedDataSet.train) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(splitedDataSet.train) //Apply normalization to the training data
        normalizer.transform(splitedDataSet.test) //Apply normalization to the test data. This is using statistics calculated from the *training* set
        return splitedDataSet
    }

    private fun showParams(model: ComputationGraph) {
        // load petal params
        val paramTable: Map<String, INDArray> = model.paramTable()
        val keys: Set<String> = paramTable.keys
        val it = keys.iterator()
        while (it.hasNext()) {
            val key = it.next()
            val values = paramTable[key]
            println("$key: " + values) //print keys
        }
    }

    private fun copyParams(fromModel: ComputationGraph, toModel: ComputationGraph) {
        // load petal params
        val paramTable: Map<String, INDArray> = fromModel.paramTable()
        val keys: Set<String> = paramTable.keys
        val it = keys.iterator()
        while (it.hasNext()) {
            val key = it.next()
            val values = paramTable[key]
//            print("$key ") //print keys
//            println(Arrays.toString(values!!.shape())) //print shape of INDArray
            setParams(toModel, key, values)
        }
        val vertexIt = fromModel.vertices.iterator()
        while (vertexIt.hasNext()) {
            val vertexName = vertexIt.next().vertexName
            val vertex = toModel.getVertex(vertexName)
            if(vertex is org.deeplearning4j.nn.graph.vertex.impl.LayerVertex) {
                println("freeze $vertexName")
                vertex.setLayerAsFrozen()
            }
        }
    }

    private fun setParams(toModel: ComputationGraph, keyToSet: String, value: INDArray?) {
        val paramTable: Map<String, INDArray> = toModel.paramTable()
        val keys: Set<String> = paramTable.keys
        val it = keys.iterator()
        var find = false
        while ( !find && it.hasNext()) {
            val key = it.next()
            if( keyToSet.equals(key) ) {
                toModel.setParam(key, value)
                find = true
            }
        }
        if(!find) println("key $keyToSet not found")
        // TODO throw exception if not find
    }

    private fun runNet(dataName: String, conf: ComputationGraphConfiguration, splitedSet: SplitTestAndTrain, useSavedModel: Boolean = true): ComputationGraph {
        val modelFilename = dataName+".zip"
        var model: ComputationGraph

        val trainingData = splitedSet.train;
        val testData = splitedSet.test;

        println("trainingData size: " + trainingData.asList().size)
        println("testData size: " + testData.asList().size)

        if( useSavedModel && File(modelFilename).exists() ) {
            println("Restoring ComputationGraph " + modelFilename)
            model = ModelSerializer.restoreComputationGraph(modelFilename) as ComputationGraph
        } else {
            model = ComputationGraph(conf)
            model.setListeners(ScoreIterationListener(100))

            println("Before training " + modelFilename)
            showParams(model)
            for (l in 0..iterations) {
                model.fit(trainingData)
            }
            println("Saving ComputationGraph into " + modelFilename)
            ModelSerializer.writeModel(model, dataName + ".zip", true)
        }
        // evaluate the network
        val eval = Evaluation()
        val output: INDArray = model.output(testData.features).first()
        eval.eval(testData.labels, output)
        println(model.summary())
        println(dataName + "Score " + eval.stats())
        return model
    }

}