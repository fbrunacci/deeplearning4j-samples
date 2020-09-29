package nyway.nd4j.samples.isic.isic2020.deotte

import krangl.*
import nyway.nd4j.samples.isic.isic2020.IDataFrameSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor

class ISICDeotteDataSet(val df : DataFrame,
                        val trainSize: Float = 0.9f,
                        val height: Long = 224L,
                        val width: Long = 224L,
                        val nchw: Boolean = true,
                        val preProcessor: DataSetPreProcessor? = null) : IDataFrameSet {

    private val nTrain: Int = (df.nrow * trainSize).toInt()
    private val nTest: Int= (df.nrow - nTrain).toInt()

    override val trainDataFrame : DataFrame
    override val testDataFrame : DataFrame

    init {
        trainDataFrame = df.take(nTrain)
        testDataFrame = df.takeLast(nTest)
    }

    override fun trainIterator(): DataSetIterator {
        return ISICDeotteDataSetIterator(trainDataFrame, height, width, nchw, preProcessor)
    }

    override fun testIterator(): DataSetIterator {
        return ISICDeotteDataSetIterator(testDataFrame, height, width, nchw, preProcessor)
    }

}
