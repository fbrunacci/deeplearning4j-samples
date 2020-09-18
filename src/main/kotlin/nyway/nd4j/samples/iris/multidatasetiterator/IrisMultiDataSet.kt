package nyway.nd4j.samples.iris.multidatasetiterator

import krangl.*
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator

class IrisMultiDataSet(val csvFile: String, val trainSize: Float = 0.9f, val batchSize :Int = 1) {

    private val df : DataFrame = DataFrame.readCSV(csvFile).shuffle()
    private val nTrain: Int
    private val nTest: Int

    init {
        nTrain = (df.nrow*trainSize).toInt()
        nTest = (df.nrow - nTrain).toInt()
    }

    fun trainIterator(): MultiDataSetIterator {
        var trainDataFrame = df.take(nTrain)
        val dataSetIterator = IrisMultiDataSetIterator(trainDataFrame, nTrain)
        return dataSetIterator
    }

    fun testIterator(): MultiDataSetIterator {
        var testDataFrame = df.takeLast(nTest)
        val dataSetIterator = IrisMultiDataSetIterator(testDataFrame, nTest)
        return dataSetIterator
    }

}

