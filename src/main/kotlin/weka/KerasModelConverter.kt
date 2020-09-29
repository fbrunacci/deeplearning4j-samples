package weka

import org.deeplearning4j.nn.conf.CNN2DFormat
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.modelimport.keras.KerasLayer
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.linalg.factory.Nd4j
import weka.dl4j.layers.lambda.CustomBroadcast
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.lang.reflect.Method
import java.nio.file.Paths
import java.util.*
import java.util.regex.Pattern

/**
 * This class loads in a folder of Keras files, and one by one converts them
 * into the native DL4J format (.zip). This is safer to work with in DL4J than
 * importing from Keras files every time, and is fine to do in this case because
 * WDL4J defines a fixed set of models - this process only needs to be done once.
 */
object KerasModelConverter {
    private var modelSummariesPath = ""
    private const val broadcastLayerRegex = "^broadcast_w(\\d+).*"
    private fun saveH5File(modelFile: File, outputFolder: File) {
        try {
            var testShape = Nd4j.zeros(1, 3, 224, 224)
            var modelName = modelFile.name
            var method: Method? = null
            try {
                method = InputType::class.java.getMethod("setDefaultCNN2DFormat", CNN2DFormat::class.java)
                method.invoke(null, CNN2DFormat.NCHW)
            } catch (ex: NoSuchMethodException) {
                System.err.println("setDefaultCNN2DFormat() not found on InputType class... " +
                        "Are you using the custom built deeplearning4j-nn.jar?")
                System.exit(1)
            }
            if (modelName.contains("EfficientNet")) {
                // Fixes for EfficientNet family of models
                testShape = Nd4j.zeros(1, 224, 224, 3)
                method!!.invoke(null, CNN2DFormat.NHWC)
                // We don't want the resulting .zip files to have 'Fixed' in the name, so we'll strip it off here
                modelName = modelName.replace("Fixed", "")
            }
            val kerasModel = KerasModelImport.importKerasModelAndWeights(modelFile.absolutePath)
            kerasModel.feedForward(testShape, false)
            // e.g. ResNet50.h5 -> KerasResNet50.zip
            modelName = "Keras" + modelName.replace(".h5", ".zip")
            val newZip = Paths.get(outputFolder.path, modelName).toString()
            kerasModel.save(File(newZip))
            println("Saved file $newZip")
        } catch (e: Exception) {
            System.err.println("""
    
    
    Couldn't save ${modelFile.name}
    """.trimIndent())
            e.printStackTrace()
        }
    }

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        if (args.size != 2) {
            System.err.println("Usage: KerasModelConverter <h5 folder path> <model summary folder path>")
            System.exit(1)
        }

        // Default location where Keras models are saved
        val modelFolderPath = args[0]
        modelSummariesPath = args[1]
        val modelFolder = File(modelFolderPath)
        val outputFolder = File(Paths.get(modelFolder.parent, "dl4j_format").toString())
        if (outputFolder.mkdir()) println("Created DL4J format folder at " + outputFolder.path)
        val modelFiles = modelFolder.listFiles() ?: throw Exception("Invalid folder name: $modelFolderPath")
        Arrays.sort(modelFiles)
        loadLambdaLayers()
        for (fileEntry in modelFiles) {
            if (fileEntry.path.endsWith(".h5")) {
                saveH5File(fileEntry, outputFolder)
            }
        }
    }

    private fun isBroadcastLayer(line: String): Boolean {
        val p = Pattern.compile(broadcastLayerRegex)
        val m = p.matcher(line)
        return m.matches()
    }

    @Throws(Exception::class)
    private fun getWidth(layerName: String): Int {
        val p = Pattern.compile(broadcastLayerRegex)
        val m = p.matcher(layerName)
        if (m.find()) {
            val width = m.group(1)
            return width.toInt()
        }
        throw Exception("Couldn't find width in layerName $layerName")
    }

    @Throws(Exception::class)
    private fun loadLambdaLayers() {
        val modelSummaries = File(modelSummariesPath).listFiles()!!
        Arrays.sort(modelSummaries)
        for (f in modelSummaries) {
            val br = BufferedReader(FileReader(f.absoluteFile))
            val modelName = f.name
            var line: String
            while (br.readLine().also { line = it } != null) {
                //          __________________________________________________________________________________________________
                // Line is~ block2c_se_expand (Conv2D)      (None, 1, 1, 144)    1008        block2c_se_reduce[0][0]
                //          __________________________________________________________________________________________________
                if (isBroadcastLayer(line)) {
                    val lineParts = line.split(" ".toRegex()).toTypedArray()
                    val layerName = lineParts[0] // -> broadcast_w65_d144_2
                    val width = getWidth(layerName)
                    KerasLayer.registerLambdaLayer(layerName, CustomBroadcast(width.toLong()))
                    println(String.format("Registered %s layer %s with width %d", modelName, layerName, width))
                }
            }
        }
    }
}