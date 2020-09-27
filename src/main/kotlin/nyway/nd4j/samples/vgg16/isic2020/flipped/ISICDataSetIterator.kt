package nyway.nd4j.samples.vgg16.isic2020.flipped

import krangl.DataFrame
import krangl.DataFrameRow
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.transform.FlipImageTransform
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

class ISICDataSetIterator(private val dataFrame: DataFrame, private val imageDir: String) : DataSetIterator {

    private val height = 224L
    private val width = 224L
    private val channels = 3L

    private var _preProcessor: DataSetPreProcessor? = null
    val labels = mutableSetOf("benign", "malignant")

    private val imageLoader = NativeImageLoader(height, width, channels)

    private var dataFrameIterator  : Iterator<DataFrameRow> = dataFrame.rows.iterator()

    override fun next(): DataSet {
        val nextRow = dataFrameIterator.next()
        val labels = rowToLabels(nextRow)
        val features = rowToImage(nextRow)
        val next = DataSet(features, labels)
        if( _preProcessor != null ) {
            preProcessor.preProcess(next)
        }
        return next
    }

    private fun rowToImage(nextRow: DataFrameRow): INDArray {
        var asWritable = imageLoader.asWritable("$imageDir/${nextRow["image_name"]}.jpg")
        val transformer = when( nextRow["flip"] ) {
            "0" -> FlipImageTransform(0)
            "-1" -> FlipImageTransform(-1)
            "1" -> FlipImageTransform(1)
            else -> null
        }
        transformer?.let { asWritable = transformer.transform(asWritable) }
        return imageLoader.asMatrix(asWritable)
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
        val featuresArray =  arrayOfNulls<INDArray>(num)
        val labelsArray = arrayOfNulls<INDArray>(num)

        for(i in 0..num-1) {
            val nextRow = dataFrameIterator.next()
            val features = imageLoader.asMatrix("$imageDir/${nextRow["image"]}.jpg")
            val labels = rowToLabels(nextRow)
            featuresArray[i] = features
            labelsArray[i] = labels
        }
        val next = MultiDataSet(featuresArray, labelsArray) as DataSet
        if( _preProcessor != null ) {
            preProcessor.preProcess(next)
        }
        return next
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

