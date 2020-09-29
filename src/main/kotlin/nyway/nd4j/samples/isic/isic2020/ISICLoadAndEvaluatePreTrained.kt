package nyway.nd4j.samples.isic.isic2020

import krangl.DataFrame
import krangl.readCSV
import nyway.nd4j.samples.Samples
import nyway.nd4j.samples.isic.isic2020.flipped.ISICDataSetIteratorFlipImpl
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import java.io.File

object ISICLoadAndEvaluatePreTrained {

    @JvmStatic
    fun main(args: Array<String>) {
        val isicFolder = "/run/media/fabien/TOSHIBA/IA/ISIC/2020"
        val imageDir = "$isicFolder/jpeg/train/"

        val testDataFrame = DataFrame.readCSV(File("${Samples.modelFolder}/vgg16/Vgg16OnISIC2020_testDataFrame.csv"))
        val testIterator = ISICDataSetIteratorFlipImpl(testDataFrame, imageDir)
        testIterator.preProcessor = VGG16ImagePreProcessor()

        val preTrainedModel = ModelSerializer.restoreComputationGraph("${Samples.modelFolder}/vgg16/Vgg16OnISIC2020.zip")
        println("> vgg16Transfer.evaluate(testIterator)")
        val eval = preTrainedModel.evaluate<Evaluation>(testIterator)
        println(eval.stats())
    }
}