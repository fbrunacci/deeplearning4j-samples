package nyway.nd4j.samples.modelimport.lambda

import org.deeplearning4j.nn.conf.CNN2DFormat
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.inputs.InputType.InputTypeConvolutional
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff

/**
 * Required for loading the EfficientNet family of models
 * Simply broadcasts the activations up to the correct size for the ElementWiseVertex to multiply the activations
 */
class CustomBroadcast : SameDiffLambdaLayer {
    private var width: Long = 0

    constructor() {}
    constructor(width: Long) {
        this.width = width
    }

    override fun defineLayer(sd: SameDiff, x: SDVariable): SDVariable {
        return x
    }

    override fun getOutputType(layerIndex: Int, inputType: InputType): InputType {
        val convolutional = inputType as InputTypeConvolutional
        val channels = convolutional.channels
        return InputType.convolutional(width, width, channels, CNN2DFormat.NHWC)
    }
}