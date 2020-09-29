package nyway.nd4j.samples.isic.isic2020.deotte

import krangl.DataFrame
import krangl.DataFrameRow
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.File

class ISICDeotteDataSetIterator(private val dataFrame: DataFrame,
                                val height: Long = 224L,
                                val width: Long = 224L,
                                val nchw: Boolean = true,
                                var _preProcessor: DataSetPreProcessor? = null ) : DataSetIterator {

    private val channels = 3L

    val labels = mutableSetOf("benign", "malignant")

    private val imageLoader = NativeImageLoader(height, width, channels)

    private var dataFrameIterator: Iterator<DataFrameRow> = dataFrame.rows.iterator()

    override fun next(): DataSet {
        val nextRow = dataFrameIterator.next()
        val labels = rowToLabels(nextRow)
        val features = rowToImage(nextRow)
        val next = DataSet(features, labels)
        _preProcessor?.let { it.preProcess(next) }
        return next
    }

    private fun rowToImage(nextRow: DataFrameRow): INDArray {
        return imageLoader.asMatrix(File("${nextRow["jpegDir"]}/${nextRow["image_name"]}.jpg"), nchw)
    }

    override fun hasNext(): Boolean {
        return dataFrameIterator.hasNext()
    }

    private fun rowToLabels(nextRow: DataFrameRow): INDArray {
        val labelsIn = DoubleArray(2)
        when (nextRow.get("benign_malignant")) {
            "benign" -> labelsIn[0] = 1.0
            "malignant" -> labelsIn[1] = 1.0
        }
        val labels = Nd4j.create(labelsIn).reshape(1, 2)
        return labels
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        return _preProcessor!!
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
        this._preProcessor = preProcessor
    }

    override fun getLabels(): List<String> {
        return labels.toList()
    }

    override fun remove() {
        TODO("Not yet implemented")
    }

    override fun next(num: Int): DataSet {
        TODO("Not yet ok")
    }

    override fun inputColumns(): Int {
        return 0
    }

    override fun totalOutcomes(): Int {
        return 0
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

    override fun batch(): Int {
        return 1
    }

}

