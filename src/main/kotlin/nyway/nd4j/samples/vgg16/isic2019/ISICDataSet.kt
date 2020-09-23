package nyway.nd4j.samples.vgg16.isic2019

import krangl.*
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor

class ISICDataSet(val imageDir: String, val csvFile: String, val trainSize: Float = 0.9f) {

    private val df : DataFrame = DataFrame.readCSV(csvFile).shuffle()
    val nTrain: Int
    val nTest: Int
    val trainDataFrame : DataFrame
    val testDataFrame : DataFrame

    init {
        nTrain = (df.nrow*trainSize).toInt()
        nTest = (df.nrow - nTrain).toInt()
        trainDataFrame = df.take(nTrain)
        testDataFrame = df.takeLast(nTest)
    }

    fun trainIterator(): DataSetIterator {
        val dataSetIterator = ISICDataSetIterator(trainDataFrame, imageDir)
        dataSetIterator.preProcessor = VGG16ImagePreProcessor()
        return dataSetIterator
    }

    fun testIterator(): DataSetIterator {
        val dataSetIterator = ISICDataSetIterator(testDataFrame, imageDir)
        dataSetIterator.preProcessor = VGG16ImagePreProcessor()
        return dataSetIterator
    }


}

