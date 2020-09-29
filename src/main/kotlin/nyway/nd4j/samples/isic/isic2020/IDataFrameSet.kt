package nyway.nd4j.samples.isic.isic2020

import krangl.DataFrame
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

interface IDataFrameSet {
    val trainDataFrame: DataFrame
    val testDataFrame: DataFrame
    fun trainIterator(): DataSetIterator
    fun testIterator(): DataSetIterator
}
