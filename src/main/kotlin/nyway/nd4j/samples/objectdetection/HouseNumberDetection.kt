/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package nyway.nd4j.samples.objectdetection

import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.helper.opencv_core
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Size
import org.datavec.api.records.metadata.RecordMetaDataImageURI
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader
import org.datavec.image.recordreader.objdetect.impl.SvhnLabelProvider
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.model.TinyYOLO
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*
import kotlin.Throws

/**
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC
 * to perform object detection with bounding boxes on The Street View House Numbers (SVHN) Dataset.
 *
 *
 * References: <br></br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br></br>
 * - The Street View House Numbers (SVHN) Dataset: http://ufldl.stanford.edu/housenumbers/ <br></br>
 *
 *
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.org/cudnn
 *
 * @author saudet
 */
object HouseNumberDetection {
    private val log = LoggerFactory.getLogger(HouseNumberDetection::class.java)

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
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        // parameters matching the pretrained TinyYOLO model
        val width = 416
        val height = 416
        val nChannels = 3
        val gridWidth = 13
        val gridHeight = 13

        // number classes (digits) for the SVHN datasets
        val nClasses = 10

        // parameters for the Yolo2OutputLayer
        val nBoxes = 5
        val lambdaNoObj = 0.5
        val lambdaCoord = 1.0
        val priorBoxes = arrayOf(doubleArrayOf(2.0, 5.0), doubleArrayOf(2.5, 6.0), doubleArrayOf(3.0, 7.0), doubleArrayOf(3.5, 8.0), doubleArrayOf(4.0, 9.0))
        val detectionThreshold = 0.5

        // parameters for the training phase
        val batchSize = 10
        val nEpochs = 20
        val learningRate = 1e-4
        val seed = 123
        val rng = Random(seed.toLong())
        val fetcher = SvhnDataFetcher()
        val trainDir = fetcher.getDataSetPath(DataSetType.TRAIN)
        val testDir = fetcher.getDataSetPath(DataSetType.TEST)
        log.info("Load data...")
        val trainData = FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng)
        val testData = FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng)
        val recordReaderTrain = ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, SvhnLabelProvider(trainDir))
        recordReaderTrain.initialize(trainData)
        val recordReaderTest = ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, SvhnLabelProvider(testDir))
        recordReaderTest.initialize(testData)

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        val train = RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true)
        train.preProcessor = ImagePreProcessingScaler(0, 1)
        val test = RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true)
        test.preProcessor = ImagePreProcessingScaler(0, 1)
        val model: ComputationGraph
        val modelFilename = "model.zip"
        if (File(modelFilename).exists()) {
            log.info("Load model...")
            model = ComputationGraph.load(File(modelFilename), true)
        } else {
            log.info("Build model...")
            val pretrained = TinyYOLO.builder().build().initPretrained() as ComputationGraph
            val priors = Nd4j.create(priorBoxes)
            val fineTuneConf = FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(Adam.Builder().learningRate(learningRate).build()) //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .build()
            model = TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections("outputs")
                    .addLayer("convolution2d_9",
                            ConvolutionLayer.Builder(1, 1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1, 1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer.Builder()
                                    .lambdaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "convolution2d_9")
                    .setOutputs("outputs")
                    .build()
            println(model.summary(InputType.convolutional(height.toLong(), width.toLong(), nChannels.toLong())))
            log.info("Train model...")
            model.setListeners(ScoreIterationListener(1))
            model.fit(train, nEpochs)
            log.info("Save model...")
            ModelSerializer.writeModel(model, modelFilename, true)
        }

        // visualize results on the test set
        val imageLoader = NativeImageLoader()
        val frame = CanvasFrame("HouseNumberDetection")
        val converter = ToMat()
        val yout = model.getOutputLayer(0) as Yolo2OutputLayer
        val labels = train.labels
        test.isCollectMetaData = true
        val colormap = arrayOf(RED, BLUE, GREEN, CYAN, YELLOW, MAGENTA, ORANGE, PINK, LIGHTBLUE, VIOLET)
        while (test.hasNext() && frame.isVisible) {
            val ds = test.next()
            val metadata = ds.exampleMetaData[0] as RecordMetaDataImageURI
            val features = ds.features
            val results = model.outputSingle(features)
            val objs = yout.getPredictedObjects(results, detectionThreshold)
            val file = File(metadata.uri)
            log.info(file.name + ": " + objs)
            val mat = imageLoader.asMat(features)
            val convertedMat = Mat()
            mat.convertTo(convertedMat, org.bytedeco.opencv.global.opencv_core.CV_8U, 255.0, 0.0)
            val w = metadata.origW * 2
            val h = metadata.origH * 2
            val image = Mat()
            opencv_imgproc.resize(convertedMat, image, Size(w, h))
            for (obj in objs) {
                val xy1 = obj.topLeftXY
                val xy2 = obj.bottomRightXY
                val label = labels[obj.predictedClass]
                val x1 = Math.round(w * xy1[0] / gridWidth).toInt()
                val y1 = Math.round(h * xy1[1] / gridHeight).toInt()
                val x2 = Math.round(w * xy2[0] / gridWidth).toInt()
                val y2 = Math.round(h * xy2[1] / gridHeight).toInt()
                opencv_imgproc.rectangle(image, Point(x1, y1), Point(x2, y2), colormap[obj.predictedClass])
                opencv_imgproc.putText(image, label, Point(x1 + 2, y2 - 2), opencv_imgproc.FONT_HERSHEY_DUPLEX, 1.0, colormap[obj.predictedClass])
            }
            frame.title = File(metadata.uri).name + " - HouseNumberDetection"
            frame.setCanvasSize(w, h)
            frame.showImage(converter.convert(image))
            frame.waitKey()
        }
        frame.dispose()
    }
}