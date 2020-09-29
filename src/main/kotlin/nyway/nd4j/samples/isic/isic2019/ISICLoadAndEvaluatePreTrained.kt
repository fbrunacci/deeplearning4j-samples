package nyway.nd4j.samples.isic.isic2019

import krangl.DataFrame
import krangl.readCSV
import nyway.nd4j.samples.Samples
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import java.io.File

object ISICLoadAndEvaluatePreTrained {

    @JvmStatic
    fun main(args: Array<String>) {
        val ISIC_DATA_DIR = "${Samples.dataFolder}/ISIC_2019"
        val imageDir = "$ISIC_DATA_DIR/ISIC_2019_Training_Input"

        val testDataFrame = DataFrame.readCSV(File("${Samples.modelFolder}/vgg16/Vgg16OnISIC_testDataFrame.csv"))
        val testIterator = ISICDataSetIterator(testDataFrame, imageDir)
        testIterator.preProcessor = VGG16ImagePreProcessor()

        val preTrainedModel = ModelSerializer.restoreComputationGraph("${Samples.modelFolder}/vgg16/Vgg16OnISIC.zip")
        println("> vgg16Transfer.evaluate(testIterator)")
        val eval = preTrainedModel.evaluate<Evaluation>(testIterator)
        println(eval.stats())
    }
}