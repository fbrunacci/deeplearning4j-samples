package nyway.nd4j.samples.iris.datasetiterator

import krangl.*
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class IrisDataSet(val csvFile: String, val trainSize: Float = 0.9f, val batchSize :Int = 1) {

    private val df : DataFrame = DataFrame.readCSV(csvFile).shuffle()
    private val nTrain: Int
    private val nTest: Int

    init {
        nTrain = (df.nrow*trainSize).toInt()
        nTest = (df.nrow - nTrain).toInt()
    }

    fun trainIterator(): DataSetIterator {
        var trainDataFrame = df.take(nTrain)
        val dataSetIterator = IrisDataSetIterator(trainDataFrame)
        return dataSetIterator
    }

    fun testIterator(): DataSetIterator {
        var testDataFrame = df.takeLast(nTest)
        val dataSetIterator = IrisDataSetIterator(testDataFrame)
        return dataSetIterator
    }

}

