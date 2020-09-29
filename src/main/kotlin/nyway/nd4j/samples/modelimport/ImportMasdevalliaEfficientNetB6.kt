package nyway.nd4j.samples.modelimport

import nyway.nd4j.samples.Samples
import nyway.nd4j.samples.modelimport.lambda.CustomBroadcast
import org.deeplearning4j.nn.modelimport.keras.KerasLayer
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport


object ImportMasdevalliaEfficientNetB6 {

    @JvmStatic
    fun main(args: Array<String>) {
        KerasLayer.registerLambdaLayer("lambda", CustomBroadcast())
//        val lambdaLayerName = listOf("broadcast_w264_d56_1", "broadcast_w264_d32_1", "broadcast_w264_d32_2", "broadcast_w132_d192_1", "broadcast_w132_d240_1", "broadcast_w132_d240_2", "broadcast_w132_d240_3", "broadcast_w132_d240_4", "broadcast_w132_d240_5", "broadcast_w66_d240_1", "broadcast_w66_d432_1", "broadcast_w66_d432_2", "broadcast_w66_d432_3", "broadcast_w66_d432_4", "broadcast_w66_d432_5", "broadcast_w33_d432_1", "broadcast_w33_d864_1", "broadcast_w33_d864_2", "broadcast_w33_d864_3", "broadcast_w33_d864_4", "broadcast_w33_d864_5", "broadcast_w33_d864_6", "broadcast_w33_d864_7", "broadcast_w33_d864_8", "broadcast_w33_d1200_1", "broadcast_w33_d1200_2", "broadcast_w33_d1200_3", "broadcast_w33_d1200_4", "broadcast_w33_d1200_5", "broadcast_w33_d1200_6", "broadcast_w33_d1200_7", "broadcast_w17_d1200_1", "broadcast_w17_d2064_1", "broadcast_w17_d2064_2", "broadcast_w17_d2064_3", "broadcast_w17_d2064_4", "broadcast_w17_d2064_5", "broadcast_w17_d2064_6", "broadcast_w17_d2064_7", "broadcast_w17_d2064_8", "broadcast_w17_d2064_9", "broadcast_w17_d2064_10", "broadcast_w17_d2064_11", "broadcast_w17_d3456_1", "broadcast_w17_d3456_2")
//        val lambdaLayerName = listOf("block1a_se_expand" , "block1b_se_expand" , "block1c_se_expand" , "block2a_se_expand" , "block2b_se_expand" , "block2c_se_expand" , "block2d_se_expand" , "block2e_se_expand" , "block2f_se_expand" , "block3a_se_expand" , "block3b_se_expand" , "block3c_se_expand" , "block3d_se_expand" , "block3e_se_expand" , "block3f_se_expand" , "block4a_se_expand" , "block4b_se_expand" , "block4c_se_expand" , "block4d_se_expand" , "block4e_se_expand" , "block4f_se_expand" , "block4g_se_expand" , "block4h_se_expand" , "block5a_se_expand" , "block5b_se_expand" , "block5c_se_expand" , "block5d_se_expand" , "block5e_se_expand" , "block5f_se_expand" , "block5g_se_expand" , "block5h_se_expand" , "block6a_se_expand" , "block6b_se_expand" , "block6c_se_expand" , "block6d_se_expand" , "block6e_se_expand" , "block6f_se_expand" , "block6g_se_expand" , "block6h_se_expand" , "block6i_se_expand" , "block6j_se_expand" , "block6k_se_expand" , "block7a_se_expand" , "block7b_se_expand" , "block7c_se_expand")
//        lambdaLayerName.forEach {
//            println(it)
//            KerasLayer.registerLambdaLayer(it, CustomBroadcast())
//        }

        val preTrainedModel = KerasModelImport.importKerasModelAndWeights(
                "${Samples.modelFolder}/masdevallia/EfficientNetB6_512x512_2019-2020_epoch12_auc_0.97.h5")

    }
}