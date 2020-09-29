package nyway.nd4j.samples.isic.isic2020.flipped

import krangl.*
import nyway.nd4j.samples.isic.isic2020.IDataFrameSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor

class ISICDataSetFlipImpl(val imageDir: String, val csvFile: String, val trainSize: Float = 0.9f) : IDataFrameSet {

    private val df : DataFrame = DataFrame.readCSV(csvFile).shuffle()

    override val trainDataFrame : DataFrame
    override val testDataFrame : DataFrame
    val nTrain: Int
    val nTest: Int

    init {
        val dataFrame = flippedDataFrame() // resizedDataFrame(10)
        trainDataFrame = dataFrame[0]
        testDataFrame = dataFrame[1]

        nTrain = trainDataFrame.rows.count()
        nTest = testDataFrame.rows.count()
    }

    private fun resizedDataFrame(ratio : Int): Array<DataFrame> {
        val malignantDataFrame = df.filter { it["benign_malignant"] eq "malignant" }
        val benignDataFrame = df.filter { it["benign_malignant"] eq "benign" }
        val benignSizedDataFrame = benignDataFrame.shuffle().take(malignantDataFrame.rows.count() * ratio)
        // merging
         bindRows(malignantDataFrame, benignSizedDataFrame).shuffle()

        val nTrain = (df.nrow*trainSize).toInt()
        val nTest = (df.nrow - nTrain).toInt()
        return arrayOf(df.take(nTrain), df.takeLast(nTest))
    }

    private fun flippedDataFrame(): Array<DataFrame> {
        val nTrain = (df.nrow*trainSize).toInt()
        val nTest = (df.nrow - nTrain).toInt()

        val testDataFrameNotFlipped = df.takeLast(nTest)
//        val trainDataFrameNotFlipped = df.take(nTrain)
        val trainDataFrameNotFlipped = df.take(nTest*10)


        val malignantDataFrame = trainDataFrameNotFlipped.filter { it["benign_malignant"] eq "malignant" }
        val benignDataFrame = trainDataFrameNotFlipped.filter { it["benign_malignant"] eq "benign" }

        val trainDataFrameWithMalignantFlipped = bindRows(
                malignantDataFrame.addColumn("flip"){""},
                malignantDataFrame.addColumn("flip"){"-1"},
                malignantDataFrame.addColumn("flip"){"1"},
                malignantDataFrame.addColumn("flip"){"0"},
                benignDataFrame.addColumn("flip"){""}
        ).shuffle()

        return arrayOf(trainDataFrameWithMalignantFlipped, testDataFrameNotFlipped)
    }

    override fun trainIterator(): DataSetIterator {
        val dataSetIterator = ISICDataSetIteratorFlipImpl(trainDataFrame, imageDir)
        dataSetIterator.preProcessor = VGG16ImagePreProcessor()
        return dataSetIterator
    }

    override fun testIterator(): DataSetIterator {
        val dataSetIterator = ISICDataSetIteratorFlipImpl(testDataFrame, imageDir)
        dataSetIterator.preProcessor = VGG16ImagePreProcessor()
        return dataSetIterator
    }


}

