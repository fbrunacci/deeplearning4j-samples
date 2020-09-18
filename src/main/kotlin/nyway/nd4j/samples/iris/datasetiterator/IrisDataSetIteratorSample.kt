package nyway.nd4j.samples.iris.datasetiterator

import nyway.nd4j.samples.iris.IrisClassifier
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File


object IrisDataSetIteratorSample {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val classLoader = this::class.java.classLoader
        val irisFile = File(classLoader.getResource("./iris.csv").path)
        val dataPetalAndSepalIn = IrisClassifier.fullMatrix

        val dataSet = IrisDataSet(irisFile.path)
        val trainIt = dataSet.trainIterator()
        val testIt = dataSet.testIterator()

        val conf = IrisClassifier.neuralNetConf(4)
        val model = MultiLayerNetwork(conf)
        model.init()
        println(model.summary())
        model.setListeners(ScoreIterationListener(100))

        for (epoch in 1..200) {
            println("epoch $epoch")
            trainIt.reset()
            while (trainIt.hasNext()) {
                val trainData = trainIt.next()
                model.fit(trainData)
            }
        }

//        val eval = Evaluation()
//        while (testIt.hasNext()) {
//            val next = testIt.next()
//            val output: INDArray = model.output(next.features) //get the networks prediction
//            eval.eval(next.labels, output) //check the prediction against the true class
//        }
        val eval = model.evaluate<Evaluation>(testIt)
        println("Score ${eval.stats()}")
    }

}