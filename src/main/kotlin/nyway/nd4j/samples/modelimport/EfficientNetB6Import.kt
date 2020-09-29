package nyway.nd4j.samples.modelimport

import nyway.nd4j.samples.Samples
import nyway.nd4j.samples.modelimport.lambda.CustomBroadcast
import org.deeplearning4j.nn.modelimport.keras.KerasLayer
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j
import weka.dl4j.scripts.keras_downloading.KerasModelConverter


object EfficientNetB6Import {

    @JvmStatic
    fun main(args: Array<String>) {
        val preTrainedModel = ModelSerializer.restoreComputationGraph("${Samples.modelFolder}/weka/KerasEfficientNetB6.zip")
        println(preTrainedModel.summary())

//        KerasLayer.registerLambdaLayer("lambda", CustomBroadcast())
//        val converterArgs = arrayOf("/home/fabien/.deeplearning4j/models/convert/", "/home/fabien/.deeplearning4j/models/convert/converted")
//        KerasModelConverter.main(converterArgs)

    }
}