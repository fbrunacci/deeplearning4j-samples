package nyway.nd4j.samples.instacart

import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.ROC
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.*
import java.net.URL

/**
 * https://www.kaggle.com/c/instacart-market-basket-analysis
 */
object InstacartMultitaskSample {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val DATA_URL = this::class.java.classLoader.getResource("instacart.tar.gz")
        val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_instacart/")

        val directory = File(DATA_PATH)
        directory.mkdir()

        val archizePath = DATA_PATH + "instacart.tar.gz"
        val archiveFile = File(archizePath)
        val extractedPath = DATA_PATH + "instacart"
        val extractedFile = File(extractedPath)

        FileUtils.copyURLToFile(DATA_URL, archiveFile)

        var fileCount = 0
        var dirCount = 0
        val BUFFER_SIZE = 4096
        val tais = TarArchiveInputStream(GzipCompressorInputStream(BufferedInputStream(FileInputStream(archizePath))))
        var entry = tais.getNextEntry()

        while(entry != null){
            if (entry.isDirectory()) {
                File(DATA_PATH + entry.getName()).mkdirs()
                dirCount = dirCount + 1
                fileCount = 0
            }
            else {
//                println("Getting ${entry.getName()}")
                val data = ByteArray(BUFFER_SIZE)
                val fos = FileOutputStream(DATA_PATH + entry.getName());
                val dest = BufferedOutputStream(fos, BUFFER_SIZE);
                var count = tais.read(data, 0, BUFFER_SIZE)

                while (count != -1) {
                    dest.write(data, 0, count)
                    count = tais.read(data, 0, BUFFER_SIZE)
                }

                dest.close()
                fileCount = fileCount + 1
            }
            if(fileCount % 1000 == 0){
                print(".")
            }
            entry = tais.getNextEntry()
        }
        val path = FilenameUtils.concat(DATA_PATH, "instacart/") // set parent directory

        val featureBaseDir = FilenameUtils.concat(path, "features") // set feature directory
        val targetsBaseDir = FilenameUtils.concat(path, "breakfast") // set label directory
        val auxilBaseDir = FilenameUtils.concat(path, "dairy") // set futures directory

        val trainFeatures = CSVSequenceRecordReader(1, ",")
        trainFeatures.initialize(NumberedFileInputSplit("$featureBaseDir/%d.csv", 1, 4000))

        val trainBreakfast = CSVSequenceRecordReader(1, ",")
        trainBreakfast.initialize(NumberedFileInputSplit("$targetsBaseDir/%d.csv", 1, 4000))

        val trainDairy = CSVSequenceRecordReader(1, ",")
        trainDairy.initialize(NumberedFileInputSplit("$auxilBaseDir/%d.csv", 1, 4000))

        val train = RecordReaderMultiDataSetIterator.Builder(20)
                .addSequenceReader("rr1", trainFeatures).addInput("rr1")
                .addSequenceReader("rr2", trainBreakfast).addOutput("rr2")
                .addSequenceReader("rr3", trainDairy).addOutput("rr3")
                .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
                .build()

        val testFeatures = CSVSequenceRecordReader(1, ",")
        testFeatures.initialize(NumberedFileInputSplit("$featureBaseDir/%d.csv", 4001, 5000))

        val testBreakfast = CSVSequenceRecordReader(1, ",")
        testBreakfast.initialize(NumberedFileInputSplit("$targetsBaseDir/%d.csv", 4001, 5000))

        val testDairy = CSVSequenceRecordReader(1, ",")
        testDairy.initialize(NumberedFileInputSplit("$auxilBaseDir/%d.csv", 4001, 5000))

        val test = RecordReaderMultiDataSetIterator.Builder(20)
                .addSequenceReader("rr1", testFeatures).addInput("rr1")
                .addSequenceReader("rr2", testBreakfast).addOutput("rr2")
                .addSequenceReader("rr3", testDairy).addOutput("rr3")
                .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
                .build()

        val conf = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dropOut(0.25)
                .updater(Adam())
                .graphBuilder()
                .addInputs("input")
                .addLayer("L1", LSTM.Builder()
                        .nIn(134).nOut(150)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10.0)
                        .activation(Activation.TANH)
                        .build(), "input")
                .addLayer("out1", RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10.0)
                        .activation(Activation.SIGMOID)
                        .nIn(150).nOut(1).build(), "L1")
                .addLayer("out2", RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10.0)
                        .activation(Activation.SIGMOID)
                        .nIn(150).nOut(1).build(), "L1")
                .setOutputs("out1", "out2")
                .build()

        val net = ComputationGraph(conf)
        net.setListeners(ScoreIterationListener(100))
        net.init()

        net.fit( train , 5);

        // Evaluate model
        val roc = ROC();
        test.reset();

        while(test.hasNext()){
            val next = test.next();
            val features =  next.features
            val output = net.output(features.first());
            roc.evalTimeSeries(next.labels.first(), output.first());
        }
        println(roc.calculateAUC());
    }
}