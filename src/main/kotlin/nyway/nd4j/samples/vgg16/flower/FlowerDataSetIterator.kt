package nyway.nd4j.samples.vgg16.flower

import nyway.nd4j.samples.Samples
import org.apache.commons.io.FileUtils
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.api.split.InputSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.common.util.ArchiveUtils
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import org.slf4j.LoggerFactory
import java.io.File
import java.io.IOException
import java.net.URL
import java.util.*

/**
 * Automatically downloads the dataset from
 * http://download.tensorflow.org/example_images/flower_photos.tgz
 * and untar's it to the users home directory
 * @author susaneraly on 3/9/17.
 */
object FlowerDataSetIterator {

    private val DATA_DIR = Samples.dataFolder
    private const val DATA_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
    private val FLOWER_DIR = "$DATA_DIR/flower_photos"
    private val allowedExtensions = BaseImageLoader.ALLOWED_FORMATS
    private val rng = Random(13)
    private const val height = 224L
    private const val width = 224L
    private const val channels = 3L
    private const val numClasses = 5
    private val log = LoggerFactory.getLogger(FlowerDataSetIterator::class.java)
    private val labelMaker = ParentPathLabelGenerator()
    private var trainData: InputSplit? = null
    private var testData: InputSplit? = null
    private var batchSize = 0

    @Throws(IOException::class)
    fun trainIterator(): DataSetIterator {
        return makeIterator(trainData)
    }

    @Throws(IOException::class)
    fun testIterator(): DataSetIterator {
        return makeIterator(testData)
    }

    @Throws(IOException::class)
    fun setup(batchSizeArg: Int, trainPerc: Int) {
        try {
            downloadAndUntar()
        } catch (e: IOException) {
            e.printStackTrace()
            log.error("IOException : ", e)
        }
        batchSize = batchSizeArg
        val parentDir = File(FLOWER_DIR)
        val filesInDir = FileSplit(parentDir, allowedExtensions, rng)
        val pathFilter = BalancedPathFilter(rng, allowedExtensions, labelMaker)
        require(trainPerc < 100) { "Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0" }
        val filesInDirSplit = filesInDir.sample(pathFilter, trainPerc.toDouble(), 100 - trainPerc.toDouble())
        trainData = filesInDirSplit[0]
        testData = filesInDirSplit[1]
    }

    @Throws(IOException::class)
    private fun makeIterator(split: InputSplit?): DataSetIterator {
        val recordReader = ImageRecordReader(height, width, channels, labelMaker)
        recordReader.initialize(split)
        val iter: DataSetIterator = RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses)
        iter.preProcessor = VGG16ImagePreProcessor()
        return iter
    }

    @Throws(IOException::class)
    fun downloadAndUntar() {
        val rootFile = File(DATA_DIR)
        if (!rootFile.exists()) {
            rootFile.mkdir()
        }
        val flowerDir = File(DATA_DIR +"/flower_photos")
        if (!flowerDir.exists()) {
            val tarFile = File(DATA_DIR, "flower_photos.tgz")
            if (!tarFile.isFile) {
                log.info("Downloading the flower dataset from $DATA_URL...")
                FileUtils.copyURLToFile(
                        URL(DATA_URL),
                        tarFile)
            }
            ArchiveUtils.unzipFileTo(tarFile.absolutePath, rootFile.absolutePath)
        }
    }
}