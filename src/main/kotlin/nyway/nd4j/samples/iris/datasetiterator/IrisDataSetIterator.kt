package nyway.nd4j.samples.iris.datasetiterator

import krangl.DataFrame
import krangl.DataFrameRow
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

class IrisDataSetIterator(private val dataFrame : DataFrame): DataSetIterator {

    private var dataFrameIterator  : Iterator<DataFrameRow> = dataFrame.rows.iterator()
    private var _preProcessor: DataSetPreProcessor? = null

    val labels = mutableSetOf("Iris-setosa", "Iris-versicolor", "Iris-virginica")

    override fun hasNext(): Boolean {
        return dataFrameIterator.hasNext()
    }

    override fun next(): DataSet {
        // TODO return next(batchSize)
        val nextRow = dataFrameIterator.next()
        val features = rowToFeatures(nextRow)
        val labels = rowToLabels(nextRow)

        val next = DataSet(features ,labels)
        if( _preProcessor != null ) {
            preProcessor.preProcess(next)
        }
        return next
    }

    override fun inputColumns(): Int {
        TODO("Not yet implemented")
    }

    override fun totalOutcomes(): Int {
        TODO("Not yet implemented")
    }

    private fun rowToLabels(nextRow: DataFrameRow): INDArray {
        val labelsIn = DoubleArray(3)
        when (nextRow.get("species")) {
            "Iris-setosa" -> labelsIn[0] = 1.0
            "Iris-versicolor" -> labelsIn[1] = 1.0
            "Iris-virginica" -> labelsIn[2] = 1.0
        }
        val labels = Nd4j.create(labelsIn).reshape(1,3)
        return labels
    }

    private fun rowToFeatures(nextRow: DataFrameRow): INDArray {
        val featuresIn = DoubleArray(4)
        featuresIn[0] = nextRow.get("sepal-length") as Double
        featuresIn[1] = nextRow.get("sepal-width") as Double
        featuresIn[2] = nextRow.get("petal-length") as Double
        featuresIn[3] = nextRow.get("petal-width") as Double
        val features = Nd4j.create(featuresIn).reshape(1,4)
        return features
    }

    override fun next(num: Int): DataSet {
        TODO("")
    }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun asyncSupported(): Boolean {
        return false
    }

    override fun reset() {
        dataFrameIterator  = dataFrame.rows.iterator()
    }

    override fun batch(): Int {
        TODO("Not yet implemented")
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor?) {
        this._preProcessor = preProcessor
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        return _preProcessor!!
    }

    override fun remove() {
        TODO("Not yet implemented")
    }


//
//    override fun hasNext(): Boolean {
//        return dataFrameIterator.hasNext()
//    }
//
//    private fun rowToINDArray(row: DataFrameRow): INDArray? {
//        val arrayOfZeros = FloatArray(9)
//        for ((i, label) in labels.withIndex()) {
//            if (row[label] == 1.0) {
//                arrayOfZeros[i] = 1f
//            }
//        }
//        return Nd4j.create(arrayOfZeros, intArrayOf(1, 9))
//    }
//
//    override fun getPreProcessor(): DataSetPreProcessor {
//        return _preProcessor!!
//    }
//
//    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
//        this._preProcessor = preProcessor
//    }
//
    override fun getLabels(): List<String> {
        return labels.toList()
    }
//
//    override fun remove() {
//        TODO("Not yet implemented")
//    }
//
//    override fun next(num: Int): DataSet {
//        val featuresArray =  arrayOfNulls<INDArray>(num)
//        val labelsArray = arrayOfNulls<INDArray>(num)
//
//        for(i in 0..num-1) {
//            val nextRow = dataFrameIterator.next()
//            val features = imageLoader.asMatrix("$imageDir/${nextRow["image"]}.jpg")
//            val labels = rowToINDArray(nextRow)
//            featuresArray[i] = features
//            labelsArray[i] = labels
//        }
//        val next = MultiDataSet(featuresArray,labelsArray) as DataSet
//        if( _preProcessor != null ) {
//            preProcessor.preProcess(next)
//        }
//        return next
//    }
//
//    override fun inputColumns(): Int {
//        return 0
//    }
//
//    override fun totalOutcomes(): Int {
//        return 0
//    }
//
//    override fun resetSupported(): Boolean {
//        return false
//    }
//
//    override fun asyncSupported(): Boolean {
//        return false
//    }
//
//    override fun reset() {
//        dataFrameIterator = dataFrame.rows.iterator()
//    }
//
//    override fun batch(): Int {
//        return 1
//    }

}

