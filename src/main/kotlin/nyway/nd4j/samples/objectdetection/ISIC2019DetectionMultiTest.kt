package nyway.nd4j.samples.objectdetection

import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.helper.opencv_core
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Size
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.records.metadata.RecordMetaDataImageURI
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.LoggerFactory
import java.io.File
import java.net.URI
import java.net.URISyntaxException
import java.util.*
import kotlin.system.exitProcess

/**
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC
 * to perform object detection with bounding boxes on images of red blood cells.
 *
 *
 * References: <br></br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br></br>
 * - Images of red blood cells: https://github.com/cosmicad/dataset <br></br>
 *
 *
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.org/cudnn
 *
 * @author saudet
 */
object ISIC2019DetectionMultiTest {
    private val log = LoggerFactory.getLogger(ISIC2019DetectionMultiTest::class.java)

    // Enable different colour bounding box for different classes
    val RED = opencv_core.RGB(255.0, 0.0, 0.0)
    val GREEN = opencv_core.RGB(0.0, 255.0, 0.0)
    val BLUE = opencv_core.RGB(0.0, 0.0, 255.0)
    val YELLOW = opencv_core.RGB(255.0, 255.0, 0.0)
    val CYAN = opencv_core.RGB(0.0, 255.0, 255.0)
    val MAGENTA = opencv_core.RGB(255.0, 0.0, 255.0)
    val ORANGE = opencv_core.RGB(255.0, 128.0, 0.0)
    val PINK = opencv_core.RGB(255.0, 192.0, 203.0)
    val LIGHTBLUE = opencv_core.RGB(153.0, 204.0, 255.0)
    val VIOLET = opencv_core.RGB(238.0, 130.0, 238.0)

    @JvmStatic
    fun main(args: Array<String>) {

        // parameters matching the pretrained TinyYOLO model
        val width = 416
        val height = 416
        val nChannels = 3
        val gridWidth = 13
        val gridHeight = 13
        val detectionThreshold = 0.1

        val model: ComputationGraph
        val modelFilename = "model_isic2019.zip"
        if (File(modelFilename).exists()) {
            log.info("Load model...")
            model = ModelSerializer.restoreComputationGraph(modelFilename)
        } else {
            log.info("Please build the model...")
            exitProcess(0)
        }

        val frame = CanvasFrame("RedBloodCellDetection")
        val imageLoader = NativeImageLoader()
        val converter = ToMat()
        val loader = NativeImageLoader(height.toLong(), width.toLong(), nChannels.toLong())
        val preProcessor = ImagePreProcessingScaler(0.0, 1.0)
        val colormap = arrayOf(RED, BLUE, GREEN, CYAN, YELLOW, MAGENTA, ORANGE, PINK, LIGHTBLUE, VIOLET)

        File("/run/media/fabien/TOSHIBA/IA/ISIC/2019/HAM10000_images_part_2").listFiles().forEach {
            //Use the nativeImageLoader to convert to numerical matrix
            val content: INDArray = loader.asMatrix(it)
            preProcessor.transform(content)

            // visualize results on the test set
            val yout = model.getOutputLayer(0) as Yolo2OutputLayer

            val results = model.outputSingle(content)
            val objs = yout.getPredictedObjects(results, detectionThreshold)
//        val file = File(metadata.uri)
//        log.info(file.name + ": " + objs)
            val mat = imageLoader.asMat(content)

            val convertedMat = Mat()
            mat.convertTo(convertedMat, org.bytedeco.opencv.global.opencv_core.CV_8U, 255.0, 0.0)
            val w = 640
            val h = 480
            val image = Mat()
            opencv_imgproc.resize(convertedMat, image, Size(w, h))
            if( objs.isNotEmpty() ) {
                val obj = objs.sortedByDescending { it.confidence }.first()
                val xy1 = obj.topLeftXY
                val xy2 = obj.bottomRightXY
                //val label = labels[obj.predictedClass]
                val label: String = ""
                val x1 = Math.round(w * xy1[0] / gridWidth).toInt()
                val y1 = Math.round(h * xy1[1] / gridHeight).toInt()
                val x2 = Math.round(w * xy2[0] / gridWidth).toInt()
                val y2 = Math.round(h * xy2[1] / gridHeight).toInt()
                opencv_imgproc.rectangle(image, Point(x1, y1), Point(x2, y2), colormap[obj.predictedClass])
                opencv_imgproc.putText(image, label, Point(x1 + 2, y2 - 2), opencv_imgproc.FONT_HERSHEY_DUPLEX, 1.0, colormap[obj.predictedClass])
            }
            frame.title = "Lesion Detection on " + it.name
            frame.setCanvasSize(w, h)
            frame.showImage(converter.convert(image))
            frame.waitKey()
        }
        frame.dispose()
     }
}