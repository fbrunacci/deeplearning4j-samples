package nyway.nd4j.samples.iris.multidatasetiterator

import krangl.DataFrame
import krangl.DataFrameRow
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.util.*


class IrisMultiDataSetIterator(private val dataFrame: DataFrame, val batchSize: Int = 135): MultiDataSetIterator {

    private var dataFrameIterator  : Iterator<DataFrameRow> = dataFrame.rows.iterator()
    private var _preProcessor: MultiDataSetPreProcessor? = null

    override fun hasNext(): Boolean {
        return dataFrameIterator.hasNext()
    }

    override fun next(): org.nd4j.linalg.dataset.api.MultiDataSet {
        return next(batchSize)
    }

    override fun next(num: Int): MultiDataSet {

        val featuresMask: Array<INDArray>? = null
        val labelMask: Array<INDArray>? = null

        val multiDataSets: ArrayList<MultiDataSet> = ArrayList()
        for(i in 0..num-1) {
            if( dataFrameIterator.hasNext() ) {
                val nextRow = dataFrameIterator.next()
                multiDataSets.add(MultiDataSet(
                        arrayOf(rowToFeatures("sepal-length","sepal-width",nextRow),
                                rowToFeatures("petal-length","petal-width",nextRow)),
                        arrayOf(rowToLabels(nextRow))
                        , featuresMask, labelMask))
            }
        }
        return MultiDataSet.merge(multiDataSets)
    }

    private fun rowToLabels(nextRow: DataFrameRow): INDArray {
        val labelsIn = DoubleArray(3)
        when (nextRow.get("species")) {
            "Iris-setosa" -> labelsIn[0] = 1.0
            "Iris-versicolor" -> labelsIn[1] = 1.0
            "Iris-virginica" -> labelsIn[2] = 1.0
        }
        val labels = Nd4j.create(labelsIn).reshape(1, 3)
        return labels
    }

    private fun rowToFeatures(col1: String, col2: String,  nextRow: DataFrameRow): INDArray {
        val featuresIn = DoubleArray(2)
        featuresIn[0] = nextRow.get(col1) as Double
        featuresIn[1] = nextRow.get(col2) as Double
        val features = Nd4j.create(featuresIn).reshape(1, 2)
        return features
    }

    private fun rowToFeaturesS(nextRow: DataFrameRow): INDArray {
        val featuresIn = DoubleArray(2)
        featuresIn[0] = nextRow.get("sepal-length") as Double
        featuresIn[1] = nextRow.get("sepal-width") as Double
        val features = Nd4j.create(featuresIn).reshape(1, 2)
        return features
    }
    private fun rowToFeaturesP(nextRow: DataFrameRow): INDArray {
        val featuresIn = DoubleArray(2)
        featuresIn[0] = nextRow.get("petal-length") as Double
        featuresIn[1] = nextRow.get("petal-width") as Double
        val features = Nd4j.create(featuresIn).reshape(1, 2)
        return features
    }

    override fun setPreProcessor(preProcessor: MultiDataSetPreProcessor?) {
        this._preProcessor = preProcessor
    }

    override fun getPreProcessor(): MultiDataSetPreProcessor {
        return _preProcessor!!
    }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun asyncSupported(): Boolean {
        return false
    }

    override fun reset() {
        dataFrameIterator = dataFrame.rows.iterator()
    }

    override fun remove() {
        TODO("Not yet implemented")
    }

}

