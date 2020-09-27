package nyway.nd4j.samples.vgg16.isic2020.deotte

import krangl.*
import nyway.nd4j.samples.vgg16.isic2020.IDataFrameSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor

class ISICDeotteDataSet(val df : DataFrame, val trainSize: Float = 0.9f) : IDataFrameSet {

    private val nTrain: Int = (df.nrow * trainSize).toInt()
    private val nTest: Int= (df.nrow - nTrain).toInt()

    override val trainDataFrame : DataFrame
    override val testDataFrame : DataFrame

    init {
        trainDataFrame = df.take(nTrain)
        testDataFrame = df.takeLast(nTest)
    }

    override fun trainIterator(): DataSetIterator {
        val dataSetIterator = ISICDeotteDataSetIterator(trainDataFrame)
        dataSetIterator.preProcessor = VGG16ImagePreProcessor()
        return dataSetIterator
    }

    override fun testIterator(): DataSetIterator {
        val dataSetIterator = ISICDeotteDataSetIterator(testDataFrame)
        dataSetIterator.preProcessor = VGG16ImagePreProcessor()
        return dataSetIterator
    }

}
